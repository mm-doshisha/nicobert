import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (BertConfig, BertForSequenceClassification,
                          BertJapaneseTokenizer, EarlyStoppingCallback,
                          PreTrainedTokenizerFast, Trainer, TrainingArguments)

from utils.loader import load_tokenizer
from utils.models import BertForTweet

sys.path.append("./twitterSNS-bert/src/")
import tokenization
from preprocess import normalizer

NUM_LABELS = 3

MAX_SEQ_LEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 20
CLS_DROPOUT = 0.1  # default
LEARNING_RATE = 2e-5
EVAL_STRATEGY = "epoch"
LOGGING_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1
REPORT_TO = "tensorboard"

N_SPLITS = 5

SEED = 2022

mode = "wiki"


model_dir = "models"

data_dir = "ft_data/wrime"


out_dir = f"models/sentiment_classification_wrime/{mode}_{NUM_LABELS}class_attention"

label_map = {"negative": 0, "positive": 1, "neutral": 2}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class TrainDataset(Dataset):
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
                "labels": self.labels[idx],
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
                "labels": self.labels[idx],
            }


def split_kfold(x, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    for tr_idx, val_idx in skf.split(x, y):
        return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
    }


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
        model_name_or_path = f"{model_dir}/twitterSNS-bert_20190311"
        tokenizer = tokenization.JapaneseTweetTokenizer(
            vocab_file=f"{model_name_or_path}/tokenizer_spm_32K.vocab.to.bert",
            model_file=f"{model_name_or_path}/tokenizer_spm_32K.model",
            normalizer=normalizer.twitter_normalizer_for_bert_encoder,
            do_lower_case=False,
        )

    train_x = df[df["Train/Dev/Test"] == "train"]["Sentence"].values
    train_y = df[df["Train/Dev/Test"] == "train"]["Avg. Readers_Sentiment"].values
    val_x = df[df["Train/Dev/Test"] == "dev"]["Sentence"].values
    val_y = df[df["Train/Dev/Test"] == "dev"]["Avg. Readers_Sentiment"].values

    train_dataset = TrainDataset(
        train_x, train_y, tokenizer, is_tweet=(mode == "twitter")
    )
    val_dataset = TrainDataset(val_x, val_y, tokenizer, is_tweet=(mode == "twitter"))

    model = BertForSequenceClassification.from_pretrained(
        f"{model_name_or_path}_transformers"
        if mode == "twitter"
        else model_name_or_path,
        num_labels=NUM_LABELS,
        classifier_dropout=CLS_DROPOUT,
        output_hidden_states=True,
    )

    train_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy=EVAL_STRATEGY,
        logging_strategy=LOGGING_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=REPORT_TO,
        load_best_model_at_end=True,
        eval_accumulation_steps=10,
    )

    trainer = Trainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
