import json
import os
import re
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import BertProcessing

from utils.implementations import BertWordPieceJapaneseTokenizer


def tok_json2vocab(tokenizer_file):
    def get_num_unused_tokens(vocab):
        unused_tokens = [k for k in vocab if re.fullmatch(r"<unused\d{1,}>", k) != None]
        unused_tokens = sorted(unused_tokens, key=lambda x: int(re.sub(r"\D", "", x)))
        num_unused_tokens = len(unused_tokens)
        for i, ut in enumerate(unused_tokens):
            ut_idx = int(re.sub(r"\D", "", ut))
            if ut_idx != i:
                raise Exception("Invalid unused token index: {ut_idx}\n")

        return num_unused_tokens

    par_dir = Path(tokenizer_file).parent
    if "vocab.txt" in os.listdir(par_dir):
        vocab_file = str(par_dir / "vocab.txt")
        with open(vocab_file, "r") as f:
            vocab = f.readlines()
        num_unused_tokens = get_num_unused_tokens(vocab)
    else:
        with open(tokenizer_file) as f:
            data = json.load(f)
        try:
            vocab_dict = data["model"]["vocab"]
        except:
            raise Exception("Invalid json structure.")
        vocab_file = str(par_dir / "vocab_generated.txt")
        with open(vocab_file, "w") as f:
            vocab_tuple = sorted(vocab_dict.items(), key=lambda x: x[1])
            for vo, _ in vocab_tuple:
                f.write(vo)
        num_unused_tokens = get_num_unused_tokens(vocab_dict.keys())

    return vocab_file, num_unused_tokens


def load_tokenizer(mode, tokenizer_file, mecab_dic_path=None):
    if mode == "sentencepiece":
        tok = Tokenizer.from_file(tokenizer_file)

        tok.post_processor = BertProcessing(
            cls=("[CLS]", tok.token_to_id("[CLS]")),
            sep=("[SEP]", tok.token_to_id("[SEP]")),
        )
    elif mode == "wordpiece":
        vocab_file, num_unused_tokens = tok_json2vocab(tokenizer_file)
        tok = BertWordPieceJapaneseTokenizer(
            vocab=vocab_file,
            num_unused_tokens=num_unused_tokens,
            mecab_dic_path=mecab_dic_path,
        )
    else:
        raise ValueError("Invalid tokenizer mode.")

    return tok
