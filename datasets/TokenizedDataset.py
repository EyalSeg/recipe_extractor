import argparse

from typing import List, Union, Callable

import torch
import pandas as pd

from tqdm import tqdm
from transformers import BertTokenizer
from transformers.tokenization_utils_base import TextInput, BatchEncoding

Tokenizer_T = Callable[[TextInput], BatchEncoding]


def create_tokenizer(
        bert_model_name: str = 'bert-base-cased', padding='max_length',  max_length=200)\
        -> Tokenizer_T:
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    closure: Tokenizer_T = \
        lambda text: tokenizer(text, padding=padding, max_length=max_length, return_tensors="pt")

    return closure


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 input_texts: List[str],
                 output_labels: Union[pd.DataFrame, torch.Tensor],
                 tokenizer: Tokenizer_T = create_tokenizer(),
                 use_tqdm=True):

        if use_tqdm:
            input_texts = tqdm(input_texts, desc="Tokenizing inputs")
        self._X: List[BatchEncoding] = [tokenizer(x) for x in input_texts]

        if isinstance(output_labels, pd.DataFrame):
            output_labels = torch.tensor(output_labels.values).to(torch.float)

        self._y: torch.Tensor = output_labels

        assert len(self._X) == len(self._y), "Number of inputs and labels do not match!"

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        X = self._X[idx]
        y = self._y[idx, :]

        return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Tokenized Dataset Creator',
        description='Takes a csv created by a scrapper and turns it into a pytorch-compatible, BERT-tokenized dataset'
        )

    parser.add_argument('--input_csv', default="../loaveandlemons_raw.csv", help="path to csv created by a scrapper")
    parser.add_argument('--output_pkl', default="../loaveandlemons_tokenized.pkl", help="Destination to save the output")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    ds = TokenizedDataset(df["text"], df[["ingredients", "instructions"]])

    torch.save(ds, args.output_pkl)

