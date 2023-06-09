import torch
import torch.nn as nn
from transformers import BertForPreTraining
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForTweet(BertForPreTraining):
    def __init__(self, config):
        super(BertForTweet, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, labels=None, attention_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_outputs = outputs[
            0
        ]  # torch.Size([batch_size, max_seq_len, hidden_size])
        cls_embeddings = sequence_outputs[
            :, 0, :
        ]  # torch.Size([batch_size, hidden_size])
        cls_embeddings = cls_embeddings.view(-1, 768)
        logits = self.classifier(self.dropout(cls_embeddings))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForNico(nn.Module):
    def __init__(self, bert_model, num_classes, dg=False):
        super(BertForNico, self).__init__()

        self.bert = bert_model

        if dg:
            self.cls = nn.Linear(in_features=768, out_features=num_classes + 1)
        else:
            self.cls = nn.Linear(in_features=768, out_features=num_classes)

        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        result = self.bert(input_ids, attention_mask=attention_mask)

        vec_0 = result[0]
        vec_0 = vec_0[:, 0, :]
        vec_0 = vec_0.view(-1, 768)
        output = self.cls(vec_0)

        return output
