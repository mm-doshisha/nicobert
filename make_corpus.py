import glob
import os
import re
import string
import unicodedata

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def remove_table_tags(text):
    bs = BeautifulSoup(text, "html.parser")
    for tag in bs.find_all("table"):
        tag.decompose()
    text = bs.get_text()
    return text


def preprocess_rev_text(text):
    text = text.replace("<br>", "\n")
    text = text.replace("<br/>", "\n")
    bs = BeautifulSoup(text, "html.parser")
    pattern = "関連|類似|外部|脚注|報道"
    tag = bs.find(re.compile("^h"), text=re.compile(pattern))
    if tag:
        sibls = [sibl for sibl in tag.next_siblings]
        tag.decompose()
        for sibl in sibls:
            if type(sibl) is bs4.element.Tag:
                sibl.decompose()
    for tag in bs.find_all("table"):
        tag.decompose()
    for tag in bs.find_all(re.compile("^h")):
        tag.decompose()
    text = bs.get_text()

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(
        r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", text
    )
    text = re.sub(r"\s+", " ", text)
    text = text.replace("・・・", " ")
    text = text.replace("...", " ")
    text = re.sub(r"(.)\1{2,}", "\g<1>\g<1>", text)
    text = text.strip()
    text = text.strip("\n")
    if "\displaystyle" in text:
        return ""
    text = text.replace("\\", "")
    text = remove_table_tags(text)
    return text


def preprocess_res_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(
        r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", text
    )

    text = re.sub(pattern, "", text)
    text = text.replace(">>", "")

    # remove mentions
    text = re.sub(r"\s+", " ", text)
    text = text.replace("・・・", " ")
    text = text.replace("...", " ")
    text = re.sub(r"(.)\1{2,}", "\g<1>\g<1>", text)
    text = text.strip()
    text = text.strip("\n")
    if "\displaystyle" in text:
        return ""
    text = text.replace("\\", "")
    text = remove_table_tags(text)
    return text


def make_corpus(mode, src_texts, output_file, min_text_length=10):
    print(f"write to {output_file}...")
    with open(output_file, "w") as f:
        for text in tqdm(src_texts):
            if mode == "rev":
                text = preprocess_rev_text(text)
            if mode == "res":
                text = preprocess_res_text(text)

            if text == "":
                continue
            period_flg = text[-1] == "。"
            for i, sent in enumerate(text.split("。")):
                sent = sent.strip()
                if i == len(text.split("。")) - 1 and not period_flg:
                    pass
                else:
                    sent += "。"
                for sent2 in sent.split("\n"):
                    sent2 = sent2.strip()
                    len_sent2 = len(sent2) - 2 if "ww" in sent2 else len(sent2)
                    if len_sent2 < min_text_length:
                        continue
                    f.write(sent2 + "\n")


def main():
    cols = ["text"]

    # src_texts = pd.read_csv("./rev_all.csv", usecols=cols).text.values
    # make_corpus("rev", src_texts, "./rev_corpus/rev_corpus.txt")

    # os.system("ls -lh ./rev_corpus/rev_corpus.txt")
    # os.system("wc -l ./rev_corpus/rev_corpus.txt")

    src_texts = pd.read_csv("./res_all.csv", usecols=cols).text.values
    make_corpus("res", src_texts, "./res_corpus/res_corpus.txt")

    os.system("ls -lh ./res_corpus/res_corpus.txt")
    os.system("wc -l ./res_corpus/res_corpus.txt")


if __name__ == "__main__":
    main()
