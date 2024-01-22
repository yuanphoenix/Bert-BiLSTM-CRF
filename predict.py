from transformers import BertTokenizer

from Model import BertNer
import config
import torch
import os

if __name__ == '__main__':
    model = BertNer().to(config.device)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "contractNerEntity1.0.pth")))
    input_str = "李丽是售货员"
    model.eval()
    inputs = tokenizer(input_str, max_length=512, padding=True, truncation=True, return_tensors="pt").to(config.device)
    logits = model(bert_input=inputs)[0]
    batch_output = model.crf.decode(logits, mask=inputs['attention_mask'].gt(0))
    print(batch_output)
