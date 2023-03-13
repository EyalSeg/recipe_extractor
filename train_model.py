import argparse

import torch
import pytorch_lightning as pl
from nebulgym.data.nebuly_dataset import NebulDataset

from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.tokenized_dataset import TokenizedDataset
from models.bert_classifier import BertClassifier


def split_dataset(dataset, ratio=0.8):
    len1 = int(len(dataset) * ratio)
    len2 = len(dataset) - len1

    return torch.utils.data.random_split(dataset, [len1, len2])


def create_balancing_sampler(dataset, batch_size):
    y = [y for X, y in dataset]
    y = torch.stack(y, dim=0)
    class_counts = y.sum(dim=0).squeeze()

    class_weights = 1 / class_counts
    labels = y.argmax(dim=1)

    weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(weights=torch.tensor(weights), num_samples=batch_size, replacement=True)
    return sampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Train The Classifier',
    )

    parser.add_argument('--dataset_file', default="loaveandlemons_tokenized.pk", help="path to the saved pytorch Dataset")
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--accelerator', default="cpu")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--output-model-path', default="model.pkl")

    args = parser.parse_args()

    classes = ["ingredients", "instructions", "filler"]

    ds = torch.load("loaveandlemons_tokenized.pkl")
    ds = NebulDataset(ds)

    train_ds, test_ds = split_dataset(ds, 0.8)
    train_ds, val_ds = split_dataset(train_ds, 0.8)

    assert len(train_ds) + len(val_ds) + len(test_ds) == len(ds)

    to_dl = lambda ds, shuffle=False, rebalance=False: DataLoader(
        ds,
        batch_size=args.batchsize,
        shuffle=shuffle,
        sampler=create_balancing_sampler(ds, args.batchsize) if rebalance else None,
    )

    train_dl = to_dl(train_ds, rebalance=True)
    val_dl = to_dl(val_ds)
    test_dl = to_dl(test_ds)

    model = BertClassifier(n_labels=len(classes), lr=5e-4, dropout=0.5)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./checkpoints',
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        accelerator=args.accelerator,
        max_epochs=args.epochs,
    )

    trainer.fit(model, train_dl, val_dl)

    trainer.test(model, dataloaders=[test_dl])

    torch.save(model, args.output_model_path)
