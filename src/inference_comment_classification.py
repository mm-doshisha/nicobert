import glob
import json
import os
import random
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from sklearn.metrics import (cohen_kappa_score, f1_score, precision_score,
                             recall_score)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BertJapaneseTokenizer, BertModel,
                          PreTrainedTokenizerFast)

from utils.datasets import VideoCommentDataset
from utils.loader import load_tokenizer
from utils.models import BertForNico

sys.path.append("./twitterSNS-bert/src/")
import tokenization
from preprocess import normalizer

base_dir = "./ft_data/nico"
MAX_SEQ_LEN = 64
batch_size = 128
SEED = 2021
num_classes = 6

mode = "nico"

if mode == "wiki":
    reward_list = [4.5]
elif mode == "twitter":
    reward_list = [5.0]
elif mode == "nico":
    reward_list = [5.5]

thresh_list = [0.5]

train_mode = "sr"

use_attention_mask = True

metric_names = ["accuracy", "precision", "recall", "f1", "kappa"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_tokenizer(mode):
    if mode == "tohoku-wwm":
        model_name_or_path = "cl-tohoku/bert-base-japanese-whole-word-masking"
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
    elif mode == "tohoku":
        model_name_or_path = "cl-tohoku/bert-base-japanese-v2"
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
    elif mode == "nico":
        model_name_or_path = "./models/runs_main/checkpoint-1200000"
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
        model_name_or_path = "./models/bert-wiki-ja_transformers"
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
        model_name_or_path = f"./models/twitterSNS-bert_20190311"
        tokenizer = tokenization.JapaneseTweetTokenizer(
            vocab_file=f"{model_name_or_path}/tokenizer_spm_32K.vocab.to.bert",
            model_file=f"{model_name_or_path}/tokenizer_spm_32K.model",
            normalizer=normalizer.twitter_normalizer_for_bert_encoder,
            do_lower_case=False,
        )

    return model_name_or_path, tokenizer


def inference(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    epoch_corrects = 0

    test_preds = []

    for batch in tqdm(dataloader):
        inputs = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]

        inputs = inputs.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            if use_attention_mask:
                outputs = model(inputs, attn_mask)
            else:
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            test_preds.append(preds)
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(dataloader.dataset)
    test_preds = torch.cat(test_preds).detach().cpu()

    return epoch_acc, test_preds


def inference_dg(model, dataloader, reward=3.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    epoch_corrects = 0

    abs_cnt = 0

    test_preds = []

    test_reservations = []
    loss = 0.0

    for batch in tqdm(dataloader):
        inputs = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]

        inputs = inputs.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            if use_attention_mask:
                outputs = model(inputs, attn_mask)
            else:
                outputs = model(inputs)

            outputs = F.softmax(outputs, dim=1)
            outputs, reservation = (
                outputs[:, :-1],
                outputs[:, -1],
            )

            test_reservations.append(reservation)
            gain = torch.gather(outputs, dim=1, index=labels.unsqueeze(1)).squeeze()
            doubling_rate = (gain.add(reservation.div(reward))).log()
            batch_loss = -doubling_rate.mean()

            loss += batch_loss / len(inputs)

            _, preds = torch.max(outputs, 1)

            test_preds.append(preds)
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(dataloader.dataset)
    test_preds = torch.cat(test_preds).detach().cpu()
    test_reservations = torch.cat(test_reservations).detach().cpu()

    return epoch_acc, test_preds, test_reservations


def softmax_response(model, dataloader, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    abs_cnt = 0
    test_preds = []

    test_max_values = []

    epoch_corrects = 0

    for batch in tqdm(dataloader):
        inputs = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]

        inputs = inputs.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            if use_attention_mask:
                outputs = model(inputs, attn_mask)
            else:
                outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)

            values, preds = torch.max(outputs, 1)
            for idx, val in enumerate(values):
                if val < thresh:
                    preds[idx] = num_classes
                    abs_cnt += 1

            test_preds.append(preds)
            test_max_values.append(values)
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(dataloader.dataset)

    epoch_abs_acc = epoch_corrects.double() / (len(dataloader.dataset) - abs_cnt)
    test_preds = torch.cat(test_preds).detach().cpu()
    test_max_values = torch.cat(test_max_values).detach().cpu()

    return test_preds, test_max_values, epoch_acc, epoch_abs_acc, abs_cnt


def calc_macros(labels_orig, preds_orig, coverage):
    cnt = int(len(labels_orig) * (100 - coverage) / 100)
    labels = labels_orig[cnt:]
    preds = preds_orig[cnt:]
    precision_avg = 0
    recall_avg = 0
    f1_avg = 0
    for c in range(num_classes):
        tp_orig = torch.sum((labels_orig == preds_orig) & (labels_orig == c))
        tp = torch.sum((labels == preds) & (labels == c))
        fp = torch.sum((labels != preds) & (preds == c))
        fn = torch.sum((labels != preds) & (labels == c))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        precision_avg += precision / num_classes
        recall_avg += recall / num_classes
        f1_avg += f1 / num_classes

    return precision_avg, recall_avg, f1_avg


def calc_scores(labels, preds, coverage):
    cnt = int(len(labels) * (100 - coverage) / 100)
    len_labels = len(labels)
    labels = labels[cnt:]
    preds = preds[cnt:]

    acc = torch.sum(labels == preds).item() / (len_labels - cnt)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    kappa = cohen_kappa_score(labels, preds)

    return {
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "kappa": kappa,
    }


def run_inference(
    model_mode,
    train_mode,
    models,
    test_dataloader,
    test_labels,
    hparam_list,
    save_fig=False,
    save_resut_to_json=False,
):
    coverages = [100, 95, 90, 85, 80, 75, 70]
    metrics = {}

    fig_save_dir = f"./figures/{train_mode}/{model_mode}"
    if save_fig and not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir, exist_ok=True)
    result_dir = f"./result_inference"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    def init_metrics():
        for k in metric_names:
            metrics[k] = {str(hparam): [] for hparam in hparam_list}

    def show():
        result = {}
        for k in metric_names:
            metrics[k][str(hparam)] = []
        for cvg in coverages:
            print(cvg)
            scores = calc_scores(sorted_test_labels, sorted_test_preds, cvg)
            tmp_result = {}
            for k, score in scores.items():
                metrics[k][str(hparam)].append(score)
                print(f"{k}: {score:.4f}")
                tmp_result[k] = score
            result[cvg] = tmp_result
        if save_resut_to_json:
            with open(f"{result_dir}/{mode}_{train_mode}.json", "w") as f:
                f.write(json.dumps(result, indent=4))

    def plot():
        for metric_name in metrics.keys():
            for hparam in hparam_list:
                plt.plot(coverages, metrics[metric_name][str(hparam)], marker=".")
            plt.xlabel("coverage")
            plt.ylabel(metric_name)
            hparam_plot_name = hparam_name if train_mode == "sr" else "o"
            plt.legend(
                [f"{hparam_plot_name}={hparam}" for hparam in hparam_list],
                loc="lower left",
                fontsize="x-small",
            )
            if save_fig:
                plt.savefig(f"{fig_save_dir}/{metric_name}.png")
            plt.close()

    if train_mode == "sr":
        print("softmax response")
        hparam_name = "thresh"

        model = models["-1"] if isinstance(models, dict) else models
        init_metrics()
        for hparam in hparam_list:
            print(f"{hparam_name}: {hparam}")
            (
                test_preds,
                test_max_values,
                epoch_acc,
                epoch_abs_acc,
                abs_cnt,
            ) = softmax_response(model, test_dataloader, thresh=hparam)

            print(f"abstention rate: {1 - (abs_cnt/len(test_preds)):.3f}")
            print(f"accuracy (all abstention): {epoch_abs_acc:.3f}")

            idxs = torch.argsort(test_max_values)
            sorted_test_preds = test_preds[idxs]
            sorted_test_labels = test_labels[idxs]
            show()
        plot()

    elif train_mode == "dg":
        print("deep gamblers")
        hparam_name = "reward"
        init_metrics()
        for hparam in hparam_list:
            print(f"reward: {reward}")
            assert isinstance(
                models, dict
            ), "DG requires pretrained models for each hpram 'o'."
            model = models[str(reward)]
            epoch_acc, test_preds, test_reservations = inference_dg(
                model, test_dataloader, reward=hparam
            )

            print(f"abstention rate: {1 - (abs_cnt/len(test_preds)):.3f}")
            print(f"accuracy (all abstention): {epoch_abs_acc:.3f}")

            idxs = torch.argsort(test_reservations, descending=True)
            sorted_test_preds = test_preds[idxs]
            sorted_test_labels = test_labels[idxs]
            show()
        plot()


if __name__ == "__main__":
    seed_all(SEED)

    model_name_or_path, tokenizer = get_tokenizer(mode)

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for reward in reward_list:
        if train_mode == "sr":
            model_path = f"./models/ft_nico/{mode}"
        elif train_mode == "dg":
            model_path = f"./models/ft_nico_dg/{mode}/reward={reward}"

        bert_model = BertModel.from_pretrained(
            f"{model_path}/model.pth",
            config=f"{model_path}/config.json",
        )
        trained_model = BertForNico(bert_model, num_classes, dg=(train_mode == "dg"))
        if device == torch.device("cpu"):
            trained_model.load_state_dict(
                torch.load(f"{model_path}/model.pth", map_location=torch.device("cpu"))
            )
        else:
            trained_model.load_state_dict(torch.load(f"{model_path}/model.pth"))

        if train_mode == "sr":
            hparam_list = thresh_list
        elif train_mode == "dg":
            hparam_list = reward_list
            if not isinstance(trained_model, dict):
                trained_model = {str(reward_list[0]): trained_model}
        run_inference(
            mode, train_mode, trained_model, test_dataloader, test_labels, hparam_list
        )
