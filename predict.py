from transformers import BertTokenizer

from Model import BertNer
import config
import torch
import os

if __name__ == '__main__':
    model = BertNer()
    config = config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "contractNerEntity.pth")))
    # while True:
    input_str = "李丽是售货员"
    model.eval()
    input = ['[CLS]'] + [item for item in input_str]
    sentence_to_id = [tokenizer.convert_tokens_to_ids(id) for id in input]
    sentence_to_id = torch.tensor(sentence_to_id, dtype=torch.long)

    sentence_mask = sentence_to_id.gt(0)
    batch_sentence_mask = [1, ] * len(input)
    batch_sentence_mask[0] = 0
    batch_sentence_mask = torch.tensor(batch_sentence_mask, dtype=torch.long)

    sentence_to_id.unsqueeze_(0)
    sentence_mask.unsqueeze_(0)
    batch_sentence_mask.unsqueeze_(0)

    logits = model(sentence_to_id, sentence_mask, batch_sentence_mask, None)[0]
    batch_label_mask = torch.tensor([1] * len(input_str), dtype=torch.bool).unsqueeze_(0)
    out = model.crf.decode(emissions=logits, mask=batch_label_mask)
