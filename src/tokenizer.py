import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def tokenizer_factory(
    tokenizer_model_name: str,
    additional_special_tokens: list[str],
) -> PreTrainedTokenizerFast:
    original_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name,
        additional_special_tokens=list(additional_special_tokens),
        use_fast=True,
    )
    return original_tokenizer


def remove_token(
    tensor: torch.Tensor, token_id: int
) -> torch.Tensor | list[torch.Tensor]:
    if tensor.dim() == 2:
        slices = []
        for slice in tensor:
            not_token_mask = slice != token_id
            slices.append(slice[not_token_mask])
        return slices
    elif tensor.dim() == 1:
        not_token_mask = tensor != token_id
        return tensor[not_token_mask]
    else:
        raise RuntimeError(
            "Function does not work with tensors with more than 2 dimensions"
        )
