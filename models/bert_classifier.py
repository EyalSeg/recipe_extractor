import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pytorch_lightning as pl

from torchmetrics import Accuracy
from transformers import BertModel

BERT_STATE_SIZE = 768


class BertClassifier(pl.LightningModule):
    def __init__(self, bert_model_name="bert-base-cased", n_labels=3, fine_tune_bert=False, lr=1e-3, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self._bert = BertModel.from_pretrained(bert_model_name)
        if not fine_tune_bert:
            self.set_bert_grad(False)

        self._post_bert = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(BERT_STATE_SIZE, n_labels),
            torch.nn.Softmax(dim=1)
        )

        self._lr = lr
        self._loss = torch.nn.CrossEntropyLoss()

        create_acc = lambda: Accuracy(task="multiclass", num_classes=n_labels).to(device)
        self._train_acc = create_acc()
        self._test_acc = create_acc()
        self._val_acc = create_acc()

    def forward(self, X):
        bert_output = self._bert(input_ids=X.input_ids.squeeze(1), attention_mask=X.attention_mask)
        y_ = self._post_bert(bert_output.pooler_output)

        return y_

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def training_step(self, train_batch, batch_idx):
        loss = self._batch_loss(train_batch, self._train_acc)

        self.log("train_loss", loss)
        self.log("train_acc", self._train_acc, on_step=True, on_epoch=True)

        return loss

    def training_step(self, val_batch, batch_idx):
        loss = self._batch_loss(val_batch, self._val_acc)

        self.log("val_loss", loss)
        self.log("val_acc", self._val_acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self._batch_loss(test_batch, self._test_acc)

        self.log("test_loss", loss)
        self.log("test_acc", self._test_acc, on_step=True, on_epoch=True)

        return loss

    def set_bert_grad(self, require_grad):
        for param in self._bert.parameters():
            param.requires_grad = require_grad

    def _batch_loss(self, batch, accuracy_obj):
        X, y = batch
        y_ = self(X)

        self._accuracy(accuracy_obj, y, y_)
        loss = self._loss(y_, y)

        return loss

    def _accuracy(self, accuracy_obj, y, y_):
        argmax = lambda tensor: torch.argmax(tensor, dim=1)

        y = argmax(y)
        y_ = argmax(y_)

        accuracy_obj(y_, y)