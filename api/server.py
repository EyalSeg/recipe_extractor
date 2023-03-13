from typing import List

import torch
from flask import Flask, request
from transformers import BatchEncoding

from datasets.tokenized_dataset import create_tokenizer
from models.bert_classifier import BertClassifier
from scrapper.scrapper import get_div_texts_from_url

app = Flask(__name__)
model = torch.load("../model.pkl")

tokenize = create_tokenizer()

classes = ["ingredients", "instructions"]


def merge_batches(batches: List[BatchEncoding]) -> BatchEncoding:
    cat = lambda tensors: torch.cat(tensors, dim=0)

    return BatchEncoding({
        "input_ids": cat([batch.input_ids for batch in batches]),
        "attention_mask": cat([batch.attention_mask for batch in batches])
    })


@app.route("/recipe", methods=["POST"])
def recipe_from_url():
    input_json = request.get_json(force=True)

    texts = get_div_texts_from_url(input_json["url"])

    tokens = [tokenize(text) for text in texts][:4]
    batch = merge_batches(tokens)
    probabilities = model(batch)

    most_probable = probabilities.argmax(dim=0)
    results = {classes[i]: texts[most_probable[i].item()] for i in range(len(classes))}

    return results
