import glob
import os
import random
import sys
import tarfile

import fugashi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchtext  # torchtextを使用
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (BertForPreTraining, BertJapaneseTokenizer, BertModel,
                          PreTrainedTokenizerFast)

from utils.datasets import VideoCommentDataset
from utils.loader import load_tokenizer
from utils.models import BertForNico

sys.path.append("./twitterSNS-bert/src/")
import tokenization
from bert_classification_inference2 import get_tokenizer, run_inference
from preprocess import normalizer

base_dir = "./ft_data/nico"
max_length = 64  # 512
MAX_SEQ_LEN = 64  # 512
batch_size = 128  # 32
num_epochs = 12
num_epochs_ce = 3  # epoch for cross entropy
SEED = 2021
num_classes = 6


mode = "nico"

if mode == "wiki":
    reward_list = [4.5]
elif mode == "twitter":
    reward_list = [5.0]
elif mode == "nico":
    reward_list = [5.5]

thresh_list = np.arange(0.1, 1.0, 0.1)

train_mode = "sr"

fig_save_dir = f"./figures/{train_mode}/{mode}"

use_attention_mask = True

save_fig = False
show_iter = False

val_size = 0.022339638592412555  # len(test_df)/len(train_df)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_model(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    for param in model.cls.parameters():
        param.requires_grad = True


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    epoch_state_dic = {
        "train": {
            "acc": [],
            "loss": [],
        },
        "val": {
            "acc": [],
            "loss": [],
        },
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print("-----start-------")

    model.to(device)

    torch.backends.cudnn.benchmark = True

    batch_size = dataloaders_dict["train"].batch_size

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1

            for batch in dataloaders_dict[phase]:
                inputs = batch["input_ids"]
                attn_mask = batch["attention_mask"]
                labels = batch["labels"]

                inputs = inputs.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if use_attention_mask:
                        outputs = model(inputs, attn_mask)
                    else:
                        outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        if iteration % 10 == 0:
                            acc = (
                                torch.sum(preds == labels.data)
                            ).double() / batch_size
                            if show_iter:
                                print(
                                    "iteration {} || loss: {:.4f} || 10iter. || accuracy: {}".format(
                                        iteration, loss.item(), acc
                                    )
                                )

                    iteration += 1

                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            epoch_state_dic[phase]["acc"].append(epoch_acc)
            epoch_state_dic[phase]["loss"].append(epoch_loss)

            print(
                "Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_acc
                )
            )

    return model, epoch_state_dic


def train_model_dg(
    model, dataloaders_dict, criterion, optimizer, num_epochs, reward=3.5
):
    epoch_state_dic = {
        "train": {
            "acc": [],
            "loss": [],
        },
        "val": {
            "acc": [],
            "loss": [],
        },
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print("-----start-------")

    model.to(device)

    torch.backends.cudnn.benchmark = True

    batch_size = dataloaders_dict["train"].batch_size

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1

            for batch in dataloaders_dict[phase]:
                inputs = batch["input_ids"]
                attn_mask = batch["attention_mask"]
                labels = batch["labels"]

                inputs = inputs.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if use_attention_mask:
                        outputs = model(inputs, attn_mask)
                    else:
                        outputs = model(inputs)

                    if epoch >= num_epochs_ce:
                        outputs = F.softmax(outputs, dim=1)
                        outputs, reservation = outputs[:, :-1], outputs[:, -1]

                        gain = torch.gather(
                            outputs, dim=1, index=labels.unsqueeze(1)
                        ).squeeze()
                        doubling_rate = (gain.add(reservation.div(reward))).log()
                        loss = -doubling_rate.mean()
                    else:
                        loss = criterion(outputs[:, :-1], labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        if iteration % 10 == 0:
                            acc = (
                                torch.sum(preds == labels.data)
                            ).double() / batch_size
                            if show_iter:
                                print(
                                    "iteration {} || loss: {:.4f} || 10iter. || accuracy: {}".format(
                                        iteration, loss.item(), acc
                                    )
                                )

                    iteration += 1

                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            epoch_state_dic[phase]["acc"].append(epoch_acc)
            epoch_state_dic[phase]["loss"].append(epoch_loss)

            print(
                "Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_acc
                )
            )

    return model, epoch_state_dic


def run_training(train_mode, model_name_or_path, dataloaders_dict, reward_list=None):
    if train_mode == "sr":
        reward_list = [-1]
    trained_models = {}

    for reward in reward_list:
        bert_model = BertModel.from_pretrained(
            f"{model_name_or_path}_transformers"
            if mode == "twitter"
            else model_name_or_path,
        )

        model = BertForNico(bert_model, num_classes, dg=(train_mode == "dg"))
        init_model(model)

        optimizer = optim.Adam(
            [
                {"params": model.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
                {"params": model.cls.parameters(), "lr": 1e-4},
            ]
        )

        criterion = nn.CrossEntropyLoss()

        if train_mode == "sr":
            trained_model, epoch_state_dic = train_model(
                model, dataloaders_dict, criterion, optimizer, num_epochs
            )
        elif train_mode == "dg":
            trained_model, epoch_state_dic = train_model_dg(
                model, dataloaders_dict, criterion, optimizer, num_epochs, reward
            )

        trained_models[str(reward)] = trained_model

    return trained_models


if __name__ == "__main__":
    seed_all(SEED)

    model_name_or_path, tokenizer = get_tokenizer(mode)

    train_df = pd.read_csv(f"{base_dir}/train.tsv", sep="\t")
    train_x, val_x, train_y, val_y = train_test_split(
        train_df.text.values,
        train_df.label.values,
        test_size=val_size,
        random_state=SEED,
    )
    train_dataset = VideoCommentDataset(
        train_x, train_y, tokenizer, MAX_SEQ_LEN, is_tweet=(mode == "twitter")
    )
    val_dataset = VideoCommentDataset(
        val_x, val_y, tokenizer, MAX_SEQ_LEN, is_tweet=(mode == "twitter")
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    trained_models = run_training(
        train_mode, model_name_or_path, dataloaders_dict, reward_list
    )

    for hparam, trained_model in trained_models.items():
        if train_mode == "sr":
            save_dir = f"./models/ft_nico/{mode}"
        elif train_mode == "dg":
            save_dir = f"./models/ft_nico_dg/{mode}/reward={hparam}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        trained_model.bert.save_pretrained(save_dir)
        torch.save(trained_model.state_dict(), f"{save_dir}/model.pth")

    print("\ninference\n")

    test_df = pd.read_csv(f"{base_dir}/RANDOM.tsv", sep="\t")
    test_labels = torch.tensor(test_df["label"].values)
    test_dataset = VideoCommentDataset(
        test_df.text.values,
        test_df.label.values,
        tokenizer,
        MAX_SEQ_LEN,
        is_tweet=(mode == "twitter"),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if train_mode == "dg":
        hparam_list = reward_list
    elif train_mode == "sr":
        hparam_list = thresh_list
    run_inference(
        mode,
        train_mode,
        trained_models,
        test_dataloader,
        test_labels,
        hparam_list,
        save_fig,
    )
