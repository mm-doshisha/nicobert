from typing import List

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from utils.trainer import train_tokenizer

corpus_dir = "all_corpus"
input_files = f"{corpus_dir}/all_corpus_sampled.txt"
output_dir = f"{corpus_dir}/sentencepiece"
num_unused_tokens = 0
vocab_size = 32000
train_args = dict(
    vocab_size=vocab_size,
)


def main():
    tok = train_tokenizer(
        "sentencepiece", input_files, output_dir, None, num_unused_tokens, **train_args
    )

    tokenizer_file = f"{corpus_dir}/sentencepiece/tokenizer.json"
    text = (
        "ニコニコ大百科とは、ニコニコ動画上での各種用語に関する解説や、ニコニコ動画上にアップされている動画についての情報をユーザが自由に記述できるサイトである。"
    )
    tok = Tokenizer.from_file(tokenizer_file)
    print(tok.encode(text).tokens)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
    )
    print(tokenizer.tokenize(text))


if __name__ == "__main__":
    main()
