from typing import List

import MeCab
import textspan
from tokenizers import NormalizedString, PreTokenizedString


class MecabPreTokenizer:
    def __init__(self, mecab_dic_path=None):
        mecab_option = "-Owakati"
        if mecab_dic_path is not None:
            mecab_option += f" -d {mecab_dic_path}"
        self.mecab = MeCab.Tagger(mecab_option)

    def custom_split(self, idx: int, text: NormalizedString) -> List[NormalizedString]:
        tokens = self.mecab.parse(str(text)).strip().split()
        token_spans = textspan.get_original_spans(tokens, str(text))
        tokens = [text[start:end] for spans in token_spans for start, end in spans]
        return tokens

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)
