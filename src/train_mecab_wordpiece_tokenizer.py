from transformers import PreTrainedTokenizerFast

from utils.loader import load_tokenizer
from utils.trainer import train_tokenizer

mecab_dic_path = "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"
mecabrc = "/usr/local/etc/mecabrc"
corpus_dir = "./all_corpus"
input_files = f"{corpus_dir}/all_corpus_sampled.txt"
output_dir = f"{corpus_dir}/wordpiece"

num_unused_tokens = 0
limit_alphabet = 6129
vocab_size = 32000  # 32768
train_args = dict(
    vocab_size=vocab_size,
    limit_alphabet=limit_alphabet,
)


def main():
    _ = train_tokenizer(
        "wordpiece",
        input_files,
        output_dir,
        mecab_dic_path,
        num_unused_tokens,
        **train_args,
    )

    tokenizer_file = f"{corpus_dir}/wordpiece/tokenizer.json"
    text = (
        "ニコニコ大百科とは、ニコニコ動画上での各種用語に関する解説や、ニコニコ動画上にアップされている動画についての情報をユーザが自由に記述できるサイトである。"
    )
    tok = load_tokenizer("wordpiece", tokenizer_file, mecab_dic_path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
    )
    print(tokenizer.tokenize(text))


if __name__ == "__main__":
    main()
