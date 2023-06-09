from tokenizers import trainers
from tokenizers.implementations import (BertWordPieceTokenizer,
                                        SentencePieceUnigramTokenizer)
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFKC, Sequence, Strip
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
from tokenizers.processors import BertProcessing

from .pre_tokenizers import MecabPreTokenizer


class BertWordPieceJapaneseTokenizer(BertWordPieceTokenizer):
    def __init__(
        self,
        vocab=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        num_unused_tokens=10,
        mecab_dic_path=None,
        wordpieces_prefix="##",
    ):
        super().__init__(
            vocab=vocab,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            wordpieces_prefix=wordpieces_prefix,
        )
        self._tokenizer.add_special_tokens(
            [f"<unused{i}>" for i in range(num_unused_tokens)]
        )

        self._tokenizer.normalizer = Sequence([NFKC(), Strip()])

        self._tokenizer.pre_tokenizer = PreTokenizer.custom(
            MecabPreTokenizer(mecab_dic_path)
        )

        parameters = {
            "model": "BertWordPieceJapanese",
            "num_unused_tokens": num_unused_tokens,
            "mecab_dic_path": mecab_dic_path,
        }

        self._parameters.update(parameters)

    @staticmethod
    def from_file(vocab: str, **kwargs):
        vocab = WordPiece.read_file(vocab)
        return BertWordPieceJapaneseTokenizer(vocab, **kwargs)

    def save(self, path, pretty=True):
        self._tokenizer.pre_tokenizer = BertPreTokenizer()  # dummy pre_tokenizer
        super().save(path, pretty)


class SentencePieceUnigramJapaneseTokenizer(SentencePieceUnigramTokenizer):
    def __init__(
        self,
        vocab=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        num_unused_tokens=10,
        replacement="▁",
        add_prefix_space=True,
    ):
        super().__init__(
            vocab=vocab,
            replacement=replacement,
            add_prefix_space=add_prefix_space,
        )

        if vocab is not None:
            sep_token_id = self._tokenizer.token_to_id(sep_token)
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = self._tokenizer.token_to_id(cls_token)
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            self._tokenizer.post_processor = BertProcessing(
                cls=(sep_token, sep_token_id), sep=(cls_token, cls_token_id)
            )

        if self._tokenizer.token_to_id(str(unk_token)) is not None:
            self._tokenizer.add_special_tokens([str(unk_token)])
        if self._tokenizer.token_to_id(str(sep_token)) is not None:
            self._tokenizer.add_special_tokens([str(sep_token)])
        if self._tokenizer.token_to_id(str(cls_token)) is not None:
            self._tokenizer.add_special_tokens([str(cls_token)])
        if self._tokenizer.token_to_id(str(pad_token)) is not None:
            self._tokenizer.add_special_tokens([str(pad_token)])
        if self._tokenizer.token_to_id(str(mask_token)) is not None:
            self._tokenizer.add_special_tokens([str(mask_token)])

        self._tokenizer.add_special_tokens(
            [f"<unused{i}>" for i in range(num_unused_tokens)]
        )

        parameters = {
            "model": "SentencePieceUnigramJapanese",
            "num_unused_tokens": num_unused_tokens,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
        self._parameters.update(parameters)

    def train(
        self,
        files,
        vocab_size=32000,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        show_progress=True,
        unk_token="[UNK]",
    ):
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size - len(special_tokens),
            special_tokens=special_tokens,
            show_progress=show_progress,
            unk_token=unk_token,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)
