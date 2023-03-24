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
import torchtext
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
from bert_classification_train2 import init_model, run_training
from preprocess import normalizer

base_dir = "./ft_data/nico"
max_length = 64
MAX_SEQ_LEN = 64
batch_size = 128
num_epochs = 12
num_epochs_ce = 3
SEED = 2021
num_classes = 6
eps = 1e-7
reward_list = np.arange(1, num_classes + eps, 0.5)

thresh_list = np.arange(0.1, 1.0, 0.1)


mode = "twitter"


train_mode = "sr"


use_attention_mask = True

save_fig = True
show_iter = False

val_size = 0.022339638592412555  # len(test_df)/len(train_df)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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

    if train_mode == "dg":
        trained_models = run_training(
            train_mode, model_name_or_path, dataloaders_dict, reward_list
        )
    elif train_mode == "sr":
        model_path = f"./models/ft_nico/{mode}"
        bert_model = BertModel.from_pretrained(
            f"{model_path}/model.pth",
            config=f"{model_path}/config.json",
        )
        trained_model = BertForNico(bert_model, num_classes, dg=(train_mode == "dg"))

        trained_model.load_state_dict(
            torch.load(f"{model_path}/model.pth", map_location=torch.device("cpu"))
        )
        trained_models = trained_model

    if train_mode == "sr":
        hparam_list = thresh_list
    elif train_mode == "dg":
        hparam_list = reward_list
    run_inference(
        mode,
        train_mode,
        trained_models,
        val_dataloader,
        torch.tensor(val_y),
        hparam_list,
        save_fig,
    )
