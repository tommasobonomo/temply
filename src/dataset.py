import json
import logging
from math import floor

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
)
from transformers import BatchEncoding, PreTrainedTokenizerFast

from settings import Config, SplitType

logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("dataset")

Batch = dict[str, torch.Tensor]


def get_dataloader(
    config: Config,
    split: SplitType,
    tokenizer: PreTrainedTokenizerFast,
    shuffled: bool = True,
) -> DataLoader:
    dataset = Seq2seqDataset(
        config=config,
        split=split,
        tokenizer=tokenizer,
        return_raw_tokens=False,
    )
    num_samples = floor(config.data_pct * len(dataset))
    if shuffled:
        sampler: Sampler = RandomSampler(dataset, num_samples=num_samples)
    else:
        sampler = SequentialSampler(Subset(dataset, range(num_samples)))

    return DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )


class Seq2seqDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: SplitType,
        tokenizer: PreTrainedTokenizerFast,
        return_raw_tokens: bool = False,
    ) -> None:
        self.split = split
        data_path = config.split_paths[split]
        self.compositional_rank = config.compositional_rank
        self.return_raw_tokens = return_raw_tokens

        with open(data_path) as f:
            self.data = [json.loads(line.strip()) for line in f]

        self.tokenizer_fn = lambda text: self.squeeze_tokenized(
            tokenizer(text, padding="max_length", return_tensors="pt")
        )

    @staticmethod
    def squeeze_tokenized(raw_tokenization_out: BatchEncoding) -> BatchEncoding:
        """Simple wrapper function that squeezes the tensors in a given BatchEncoding if their dimension is bigger than 1"""
        if all(tens.dim() > 1 for tens in raw_tokenization_out.values()):
            squeezed_tokenization = BatchEncoding(
                data={
                    tensor_name: tensor.squeeze()
                    for tensor_name, tensor in raw_tokenization_out.items()
                }
            )
        else:
            squeezed_tokenization = raw_tokenization_out
        return squeezed_tokenization

    def __getitem__(self, index: int) -> Batch:
        raw_sample = self.data[index]

        if self.compositional_rank > 0:
            compositional_prefix = (
                " ".join(
                    f"<{ct}>"
                    for ct in raw_sample["compositional_tokens"][
                        : self.compositional_rank
                    ]
                )
                + " "
            )
            source_sequence, target_sequence = (
                compositional_prefix + raw_sample["source_sequence"],
                compositional_prefix + raw_sample["target_sequence"],
            )
        else:
            source_sequence, target_sequence = (
                raw_sample["source_sequence"],
                raw_sample["target_sequence"],
            )
        source_sequence_tokenized = self.tokenizer_fn(source_sequence)
        target_sequence_tokenized = self.tokenizer_fn(target_sequence)

        labels = torch.where(
            target_sequence_tokenized["attention_mask"].to(torch.bool),
            target_sequence_tokenized["input_ids"],
            -100,
        )

        if self.return_raw_tokens:
            return {
                **source_sequence_tokenized,
                "labels": labels,
                "raw_tokens": raw_sample["tokens"],
            }
        else:
            return {**source_sequence_tokenized, "labels": labels}

    def __len__(self):
        return len(self.data)


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: SplitType,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.split = split
        data_path = config.split_paths[split]

        with open(data_path) as f:
            self.data = [json.loads(line.strip()) for line in f]

        self.tokenizer_fn = lambda text: self.squeeze_tokenized(
            tokenizer(
                text,
                is_split_into_words=True,
                padding="longest",
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        )
        self.decode_batch = lambda input_ids: tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(config.class2id_path) as f:
            self.class2id = json.load(f)

    @staticmethod
    def squeeze_tokenized(raw_tokenization_out: BatchEncoding) -> BatchEncoding:
        """Simple wrapper function that squeezes the tensors in a given BatchEncoding if their dimension is bigger than 1"""
        if all(tens.dim() > 1 for tens in raw_tokenization_out.values()):
            squeezed_tokenization = BatchEncoding(
                data={
                    tensor_name: tensor.squeeze()
                    for tensor_name, tensor in raw_tokenization_out.items()
                }
            )
        else:
            squeezed_tokenization = raw_tokenization_out
        return squeezed_tokenization

    def __getitem__(self, index: int) -> dict:
        raw_sample = self.data[index]
        return raw_sample

    def collate_fn(self, samples: list[dict]) -> Batch:
        # Tokenize sentence
        tokenized_sequence = self.tokenizer_fn(
            [raw_sample["tokens"] for raw_sample in samples]
        )

        # Keep batch dimension if len(samples) == 1
        if len(samples) == 1:
            for key, value in tokenized_sequence.items():
                tokenized_sequence[key] = value.unsqueeze(0)

        raw_labels = []
        for sample, attention_mask, offset_mapping in zip(
            samples,
            tokenized_sequence.attention_mask,
            tokenized_sequence.offset_mapping,
        ):
            # Token classification target
            sample_labels = torch.where(
                attention_mask.to(torch.bool),
                self.class2id["O"],
                -100,
            )

            # Set original tokens labels with their class
            og_tokens_index = torch.logical_and(
                offset_mapping[:, 0] == 0, offset_mapping[:, 1] != 0
            )

            sample_labels[og_tokens_index] = torch.tensor(
                [self.class2id[label] for label in sample["labels"]]
            )

            raw_labels.append(sample_labels)

        labels = torch.stack(raw_labels)

        return {
            "input_ids": tokenized_sequence.input_ids,
            "attention_mask": tokenized_sequence.attention_mask,
            "offset_mapping": tokenized_sequence.offset_mapping,
            "labels": labels,
        }

    def __len__(self):
        return len(self.data)
