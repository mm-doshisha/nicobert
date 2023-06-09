import os

from tokenizers.implementations import SentencePieceUnigramTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer

from .implementations import (BertWordPieceJapaneseTokenizer,
                              SentencePieceUnigramJapaneseTokenizer)


def train_tokenizer(
    mode, input_files, output_dir, mecab_dic_path, num_unused_tokens, **kwargs
):
    if mode == "wordpiece":
        tokenizer = BertWordPieceJapaneseTokenizer(
            num_unused_tokens=num_unused_tokens,
            mecab_dic_path=mecab_dic_path,
        )
    elif mode == "sentencepiece":
        tokenizer = SentencePieceUnigramJapaneseTokenizer(
            num_unused_tokens=num_unused_tokens,
        )
    else:
        raise ValueError("Invalid tokenizer mode.")

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # default
    special_tokens += [f"<unused{i}>" for i in range(num_unused_tokens)]

    tokenizer.train(
        files=input_files,
        special_tokens=special_tokens,
        **kwargs,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)  # save vocab file

    tokenizer.save(f"{output_dir}/tokenizer.json")  # save json file

    return tokenizer
