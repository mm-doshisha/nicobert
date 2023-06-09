import torch
from torch.utils.data import Dataset


class VideoCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len, is_tweet=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_tweet = is_tweet

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.is_tweet:
            words = ["[CLS]"] + self.tokenizer.tokenize(self.texts[idx]) + ["[SEP]"]
            attn_mask = len(words) * [1]
            remain_len = self.max_seq_len - len(words)
            if remain_len >= 0:
                words += remain_len * ["<pad>"]
                attn_mask += remain_len * [0]
            else:
                words = words[: self.max_seq_len]
                attn_mask = attn_mask[: self.max_seq_len]
            return {
                "input_ids": torch.LongTensor(
                    self.tokenizer.convert_tokens_to_ids(words)
                ),
                "attention_mask": torch.LongTensor(attn_mask),
                "labels": self.labels[idx],
            }
        else:
            encoded_inputs = self.tokenizer(
                self.texts[idx],
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded_inputs["input_ids"][0],
                "attention_mask": encoded_inputs["attention_mask"][0],
                "labels": self.labels[idx],
            }
