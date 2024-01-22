import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
import config


class BertNer(nn.Module):
    def __init__(self):
        super(BertNer, self).__init__()
        self.dropout = nn.Dropout()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.bilstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=config.last_state_dim,
                              hidden_size=config.last_state_dim // 2, batch_first=True)
        self.crf = CRF(num_tags=len(config.ner_classes_list), batch_first=True)
        self.classifier = nn.Linear(config.last_state_dim, len(config.ner_classes_list))

    def forward(self, bert_input, batch_labels: None = None):
        attention_mask = bert_input['attention_mask']

        bert_output = self.bert(**bert_input).last_hidden_state
        lstm_output, _ = self.bilstm(bert_output)
        logits = self.classifier(lstm_output)
        # bert_output = self.bert(input_ids=batch_sentence_id, attention_mask=batch_mask).last_hidden_state
        # origin_sequence_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in
        #                           zip(bert_output, batch_sentence_mask)]
        # padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # padded_sequence_output = self.dropout(padded_sequence_output)
        # lstm_output, _ = self.bilstm(padded_sequence_output)
        # logits = self.classifier(lstm_output)
        outputs = (logits,)
        if batch_labels is not None:
            # loss_mask = batch_labels.gt(-1)
            loss = self.crf(emissions=logits, tags=batch_labels, mask=attention_mask.gt(0)) * -1
            outputs = (loss,) + outputs
        return outputs
