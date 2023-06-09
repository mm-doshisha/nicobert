import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from matplotlib import use
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             mean_absolute_error, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertJapaneseTokenizer,
                          EarlyStoppingCallback, PreTrainedTokenizerFast,
                          Trainer, TrainingArguments)

from utils.loader import load_tokenizer
from utils.models import BertForTweet

sys.path.append("./twitterSNS-bert/src/")
import tokenization
from preprocess import normalizer

NUM_LABELS = 3

MAX_SEQ_LEN = 256
BATCH_SIZE = 32
CLS_DROPOUT = 0.1  # default

SEED = 2022

use_attention_mask = True

mode = "wiki"


model_dir = "models"
data_dir = "ft_data/wrime"
if use_attention_mask:
    input_dir = (
        f"models/sentiment_classification_wrime/{mode}_{NUM_LABELS}class_attention"
    )
else:
    input_dir = f"models/sentiment_classification_wrime/{mode}_{NUM_LABELS}class"

label_map = {"negative": 0, "positive": 1, "neutral": 2}


ckpt_max_or_min = "max"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def detect_checkpoint(base_dir, max_or_min="max"):
    checkpoints = [
        ckpt for ckpt in os.listdir(base_dir) if ckpt.startswith("checkpoint-")
    ]
    idx = -1 if max_or_min == "max" else 0
    return sorted(checkpoints, key=lambda s: int(s.split("-")[-1]))[idx]


class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, is_tweet=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.is_tweet = is_tweet

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.is_tweet:
            words = ["[CLS]"] + self.tokenizer.tokenize(self.texts[idx]) + ["[SEP]"]
            attn_mask = len(words) * [1]
            remain_len = MAX_SEQ_LEN - len(words)
            if remain_len >= 0:
                words += remain_len * ["<pad>"]
                attn_mask += remain_len * [0]
            else:
                words = words[:MAX_SEQ_LEN]
                attn_mask = attn_mask[:MAX_SEQ_LEN]
            return {
                "input_ids": torch.LongTensor(
                    self.tokenizer.convert_tokens_to_ids(words)
                ),
                "attention_mask": torch.LongTensor(attn_mask),
            }
        else:
            encoded_inputs = self.tokenizer(
                self.texts[idx],
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded_inputs["input_ids"][0],
                "attention_mask": encoded_inputs["attention_mask"][0],
            }


def calc_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    mae = mean_absolute_error(labels, preds)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "MAE": mae,
        "QWK": qwk,
    }


def inference(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    test_preds = []

    for batch in tqdm(dataloader):
        inputs = batch["input_ids"]
        attn_mask = batch["attention_mask"]

        inputs = inputs.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            if use_attention_mask:
                outputs = model(inputs, attn_mask)
            else:
                outputs = model(inputs)

            preds = outputs.logits.argmax(-1)
            test_preds.append(preds)

    test_preds = torch.cat(test_preds).detach().cpu()
    return test_preds


def main():
    seed_all(SEED)

    df = pd.read_csv(f"{data_dir}/wrime_simple.tsv", sep="\t")
    if NUM_LABELS <= 3:

        def round(x):
            if x == 2:
                return 1
            elif x == -2:
                return -1
            else:
                return x

        df["Avg. Readers_Sentiment"] = df["Avg. Readers_Sentiment"].apply(round)
        if NUM_LABELS == 2:
            df = df[df["Avg. Readers_Sentiment"] != 0].reset_index(drop=True)
            df["Avg. Readers_Sentiment"] = df["Avg. Readers_Sentiment"].apply(
                lambda x: 0 if x == -1 else x
            )
        else:
            df["Avg. Readers_Sentiment"] = df["Avg. Readers_Sentiment"].apply(
                lambda x: x + 1
            )
    elif NUM_LABELS == 5:
        df["Avg. Readers_Sentiment"] = df["Avg. Readers_Sentiment"].apply(
            lambda x: x + 2
        )

    if mode == "nico":
        model_name_or_path = f"{model_dir}/runs_main/checkpoint-1200000"
        tok = load_tokenizer(
            "sentencepiece",
            f"{model_name_or_path}/tokenizer.json",
            None,
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="[UNK]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )
    elif mode == "wiki":
        model_name_or_path = f"{model_dir}/bert-wiki-ja_transformers"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{model_name_or_path}/tokenizer.json",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )
    elif mode == "twitter":
        model_name_or_path = f"{model_dir}/twitterSNS-bert_20190311_transformers"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{model_name_or_path}/tokenizer.json",
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    test_x = df[df["Train/Dev/Test"] == "test"]["Sentence"].values
    test_labels = df[df["Train/Dev/Test"] == "test"]["Avg. Readers_Sentiment"].values
    test_dataset = TestDataset(
        test_x, test_labels, tokenizer, is_tweet=(mode == "twitter")
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    checkpoint = detect_checkpoint(input_dir, ckpt_max_or_min)
    print(f"detected checkpoint: {checkpoint}")

    if mode == "twitter":
        model = BertForSequenceClassification.from_pretrained(
            f"{input_dir}/{checkpoint}",
            num_labels=NUM_LABELS,
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            f"{input_dir}/{checkpoint}",
            num_labels=NUM_LABELS,
        )

    test_preds = inference(model, test_dataloader)
    metrics = calc_metrics(test_labels, test_preds)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
