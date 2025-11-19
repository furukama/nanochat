"""
PleIAs SYNTH dataset.
Each row pairs a rich user query with model-generated reasoning + answers.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from datasets import load_dataset

from tasks.common import Task

# Cache HF dataset objects so multiple task instances don't redownload the shards.
_SYNTH_DATASETS = {}


def _get_synth_dataset(split: str):
    if split not in _SYNTH_DATASETS:
        _SYNTH_DATASETS[split] = load_dataset("PleIAs/SYNTH", split=split)
    return _SYNTH_DATASETS[split]


class PleiasSynth(Task):
    """
    Task wrapper around PleIAs/SYNTH.

    Args:
        split: HF split to load (dataset only ships a 'train' split today).
        languages: Optional iterable of ISO codes to keep, e.g. ("en",).
        include_seed_text: Append the seed reference text to the user prompt.
        include_reasoning: Prepend synthetic reasoning before the final answer.
        max_examples: Hard stop to cap dataset length for smoke tests.
    """

    def __init__(
        self,
        split: str = "train",
        languages: Optional[Iterable[str]] = None,
        include_seed_text: bool = False,
        include_reasoning: bool = True,
        max_examples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(split, str) and split, "split must be a non-empty string"
        self.split = split
        if languages is None:
            self.languages: Optional[Sequence[str]] = None
        else:
            if isinstance(languages, str):
                languages = [languages]
            languages = list(languages)
            assert len(languages) > 0, "languages iterable must be non-empty"
            self.languages = tuple(sorted(languages))
        self.include_seed_text = include_seed_text
        self.include_reasoning = include_reasoning

        dataset = _get_synth_dataset(split)
        indices: List[int] = []
        allowed_langs = set(self.languages) if self.languages else None
        for idx in range(len(dataset)):
            row = dataset[idx]
            lang = row.get("language", "").strip()
            if allowed_langs and lang not in allowed_langs:
                continue
            indices.append(idx)
            if max_examples is not None and len(indices) >= max_examples:
                break

        assert len(indices) > 0, f"No SYNTH rows matched the filters for split={split}"
        self.dataset = dataset
        self.indices = indices
        self.length = len(indices)

    # -------------------------------------------------------------------------
    def num_examples(self):
        return self.length

    def _normalize_field(self, row, key: str):
        value = row.get(key, "")
        if value is None:
            return ""
        assert isinstance(value, str), f"{key} must be a string"
        return value.strip()

    def get_example(self, index):
        assert 0 <= index < self.length, f"Index {index} out of range"
        row = self.dataset[self.indices[index]]

        query = self._normalize_field(row, "query")
        answer = self._normalize_field(row, "synthetic_answer")
        assert query, "Empty SYNTH query"
        assert answer, "Empty SYNTH synthetic_answer"

        user_chunks = [query]
        constraints = self._normalize_field(row, "constraints")
        if constraints:
            user_chunks.append(f"Constraints:\n{constraints}")
        if self.include_seed_text:
            seed = self._normalize_field(row, "query_seed_text")
            if seed:
                user_chunks.append(f"Seed text:\n{seed}")

        assistant_chunks: List[str] = []
        if self.include_reasoning:
            reasoning = self._normalize_field(row, "synthetic_reasoning")
            if reasoning:
                assistant_chunks.append(reasoning)
        assistant_chunks.append(answer)

        conversation = {
            "messages": [
                {"role": "user", "content": "\n\n".join(user_chunks)},
                {"role": "assistant", "content": "\n\n".join(assistant_chunks)},
            ]
        }
        return conversation

