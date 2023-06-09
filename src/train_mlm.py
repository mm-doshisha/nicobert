import dataclasses
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing
from transformers import (AutoTokenizer, BertConfig, BertForMaskedLM,
                          BertForPreTraining, DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask, HfArgumentParser,
                          LineByLineTextDataset, PreTrainedTokenizerFast,
                          Trainer, TrainingArguments, pipeline)
from transformers.trainer_utils import get_last_checkpoint

from utils.args import DataTrainingArguments, ModelArguments
from utils.loader import load_tokenizer
from utils.logger import get_logger_from_yaml, get_root_logger

SEED = 2021


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_dataclass_from_json(dtype, json_file):
    assert dataclasses.is_dataclass(dtype)
    with open(json_file) as f:
        data = json.load(f)
    keys = {f.name for f in dataclasses.fields(dtype) if f.init}
    inputs = {k: v for k, v in data.items() if k in keys}
    obj = dtype(**inputs)
    return obj


def main():
    seed_all(SEED)

    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--training")
    parser.add_argument("--tokenizer")
    config_file_args = parser.parse_args()
    model_config_file = config_file_args.model
    data_config_file = config_file_args.data
    training_config_file = config_file_args.training
    tokenizer_config_file = config_file_args.tokenizer

    ################
    # set arguments
    ################
    with open(tokenizer_config_file) as f:
        data = json.load(f)
        tokenizer_mode = data["tokenizer_mode"]
        tokenizer_file = data["tokenizer_file"]
        mecab_dic_path = data.get("mecab_dic_path", None)

    data_args = load_dataclass_from_json(DataTrainingArguments, data_config_file)
    model_config = BertConfig.from_json_file(model_config_file)

    training_args = load_dataclass_from_json(TrainingArguments, training_config_file)

    #############
    # set logger
    #############

    logger = get_logger_from_yaml("config/log_config.yml")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    ####################
    # detect checkpoint
    ####################

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #################
    # create raw dataset
    #################

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        if isinstance(data_args.train_file, list):
            ext = data_args.train_file[0].split(".")[-1]
        else:
            ext = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        if isinstance(data_args.validation_file, list):
            ext = data_args.validation_file[0].split(".")[-1]
        else:
            ext = data_args.validation_file.split(".")[-1]
    if ext == "txt":
        ext = "text"
    raw_datasets = load_dataset(ext, data_files=data_files)

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            ext,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            ext,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
        )

    ############################
    # load pretrained tokenizer
    ############################

    tok = load_tokenizer(tokenizer_mode, tokenizer_file, mecab_dic_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )

    #############
    # load model
    #############

    model = BertForMaskedLM(model_config)

    ######################
    # preprocess datasets
    ######################

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(
            local=True, desc="dataset map tokenization"
        ):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    ####################
    # set data collator
    ####################

    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )
    if data_args.whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    ##############
    # set trainer
    ##############

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    ###############
    # run training
    ###############

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None:
            max_train_samples = min(max_train_samples, data_args.max_train_samples)
        metrics["train_samples"] = max_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
