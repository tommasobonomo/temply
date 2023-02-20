import logging
from functools import partial
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding

from settings import Config, SplitType

logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("dataset")


# Batch is a tuple of (id, tokenized sentence, int label)
Batch = Tuple[torch.Tensor, BatchEncoding, torch.Tensor]

label2idx = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16,
}
idx2label = {value: key for key, value in label2idx.items()}


class PosDataset(Dataset):
    def __init__(self, config: Config, split: SplitType) -> None:
        super().__init__()
        self.split = split

        tokenizer = AutoTokenizer.from_pretrained(config.model.transformer_model)
        self.tokenizer_fn = partial(
            tokenizer,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Read corresponding data split
        data_path = config.split_paths[split]

        self.data = self._read_data_file(data_path)

    @staticmethod
    def _read_data_file(path: Path) -> List[List[Tuple[str, str]]]:
        sentences = []
        sentence = []
        with open(path) as f:
            for raw_line in f.readlines():
                line = raw_line.strip()
                cells = line.split("\t")
                if len(cells) == 2:
                    word, tag = cells
                    sentence.append((word, tag))
                else:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    sentence = []

        if len(sentence) > 0:
            sentences.append(sentence)

        return sentences

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Batch:
        # Extract words and labels from required sentence
        sentence = self.data[index]
        words, labels = zip(*sentence)

        # Convert labels to fixed int value, determined in `label2idx`
        tensor_labels = torch.tensor(
            [label2idx[label] for label in labels], dtype=torch.long
        )

        # Tokenize words and squeeze results (we don't need a prepended batch dimension of 1)
        tokenization_result = self.tokenizer_fn(words)
        for key, value in tokenization_result.items():
            tokenization_result[key] = torch.squeeze(value)

        # Build and align final label tensor
        label_offset = tokenization_result["offset_mapping"]
        aligned_labels = torch.full((len(label_offset),), -100, dtype=torch.long)
        idx_original_labels = (label_offset[:, 0] == 0) & (label_offset[:, 1] != 0)

        # Check that we can align our labels with no issues
        if tensor_labels.size(0) != idx_original_labels.sum():
            console_logger.warn(
                f"There has been an error in encoding labels for index {index} of split {self.split}."
                + "Maybe a data check should be carried out? Will return empty-labelled tensor."
            )
        else:
            # Align the labels tensor
            aligned_labels[idx_original_labels] = tensor_labels

        return torch.tensor(index), tokenization_result, aligned_labels
