from typing import List, Tuple

import pytorch_lightning as pl
import torch
from transformers import AdamW, AutoModel

from settings import Config
from src.dataset import Batch, label2idx


class PosModel(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.learning_rate = config.model.learning_rate

        # Load pretrained transformer encoder with Huggingface
        self.transformer_encoder = AutoModel.from_pretrained(
            config.model.transformer_model
        )

        # Define feedforward layer to classify tokens
        self.classification_layer = torch.nn.Linear(
            in_features=self.transformer_encoder.config.hidden_size,
            out_features=len(label2idx),
            bias=config.model.use_bias,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        self.save_hyperparameters()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, batch: Batch) -> List[Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        indexes, sentences, _ = batch
        # Use offset mapping to return predicted labels with the same length of un-tokenized input sentences
        offset_mapping = sentences.pop("offset_mapping")
        # Run sentences through selected transformer encoder.
        # The output shape is (B, L, H): Batch size, Length of sequence, Hidden size of encoder output
        encoded_sentences = self.transformer_encoder(**sentences).last_hidden_state

        # Pass through classification layer, that for each token determines what class describes it most
        predicted_logits = self.classification_layer(encoded_sentences)
        predicted_tags = torch.argmax(predicted_logits, dim=-1)

        predicted_int_tags = []
        for sample_index, all_tags, token_offset in zip(
            indexes, predicted_tags, offset_mapping
        ):
            idx_original_tokens = (token_offset[:, 0] == 0) & (token_offset[:, 1] != 0)
            predicted_int_tags.append(
                (
                    sample_index.cpu().tolist(),
                    all_tags[idx_original_tokens].cpu().tolist(),
                )
            )

        return predicted_int_tags

    def step(self, batch: Batch) -> torch.Tensor:
        _, sentences, labels = batch
        # Remove not needed attribute
        sentences.pop("offset_mapping")
        # Run sentences through selected transformer encoder.
        # The output shape is (B, L, H): Batch size, Length of sequence, Hidden size of encoder output
        encoded_sentences = self.transformer_encoder(**sentences).last_hidden_state

        # Pass through classification layer, that for each token determines what class describes it most
        predicted_logits = self.classification_layer(encoded_sentences)

        # We're using a standard cross-entropy loss function between the predicted probablities for a token
        # to be in each class and the integer labels.
        # PyTorch intelligently ignores labels set as `-100` so we don't need to worry about attention masks.
        # We accumulate the final loss for the batch as the mean loss over each sentence followed by the
        # mean over all sentences.
        # Torch requires the class dimension to be the one straight after the batch dimension, so we need to
        # swap the axes of the predicted logits
        predicted_logits = torch.swapaxes(predicted_logits, 1, 2)
        loss = self.loss_fn(predicted_logits, labels)
        return loss

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:  # type: ignore
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
