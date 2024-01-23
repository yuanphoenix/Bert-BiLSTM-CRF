import copy
import os

import torch.nn
from torch.utils.data import Dataset
from transformers import BertTokenizer

import config

tokenizer = BertTokenizer.from_pretrained(config.bert_path)


def collate_fn(batch):
    """
    :param batch:一批次的数据
    :return:
    """
    sentence_list = [x[0] for x in batch]
    labels_list = [x[1] for x in batch]

    max_len = 0
    for x in sentence_list:
        max_len = max(max_len, len(x))
    if max_len > 510:
        max_len = 510
        assert len(sentence_list) == len(labels_list)
        for i in range(len(sentence_list)):
            sentence_list[i] = sentence_list[i][:510]
            labels_list[i] = labels_list[i][:510]
    max_len += 2

    # 现在开始填充
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_output = []
    for x, y in zip(sentence_list, labels_list):
        input_id = [tokenizer.cls_token_id] + x + [tokenizer.sep_token_id]  # 添加 [CLS]和[SEP]
        token_type_id = [0] * max_len
        attention = [1] * len(input_id) + [0] * (max_len - len(input_id))
        label = [config.txt2label[tokenizer.cls_token]] + y + [config.txt2label[tokenizer.sep_token]] + [
            config.txt2label[tokenizer.pad_token]] * (max_len - len(input_id))  # 增加标签
        input_id.extend([0] * (max_len - len(input_id)))

        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_mask.append(attention)
        label_output.append(label)

    input_ids = torch.tensor(input_ids, device=config.device, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, device=config.device, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, device=config.device, dtype=torch.long)
    label_output = torch.tensor(label_output, device=config.device, dtype=torch.long)

    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, "attention_mask": attention_mask}, label_output


class NerDataSet(Dataset):
    def __init__(self, mode: str):
        self.max_length = config.max_length
        self.lines = []
        self.flags = []
        self._line = []
        self._flag = []
        self.data = self.process(mode)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def process(self, mode: str):
        """
        :param mode: 要训练的模式
        :return: data 是一个列表，元素是二元组。第其中一个元素是 句子id 和句子掩码。第二个元素是单词对应的flag的集合
        """
        data = []
        data_dir = os.path.join(os.getcwd(), "data")

        with open(os.path.join(data_dir, f"{mode}.txt"), mode='r', encoding='utf-8') as f:
            file_content = f.readlines()
            for line in file_content:
                if len(line) <= 1:
                    # 如果这一行是空的，那么说明这句话到头了
                    self.lines.append(copy.deepcopy(self._line))
                    self.flags.append(copy.deepcopy(self._flag))
                    self._line.clear()
                    self._flag.clear()
                else:
                    word, flag = line.split(' ', 1)
                    self._line.append(word)
                    self._flag.append(flag.strip())
            sentences = []
            sentence_labels = []
            for word_list in self.lines:
                # 原来的汉字被转化为了token_id
                sentence_to_id = [tokenizer.convert_tokens_to_ids(word) for word in word_list]
                sentences.append(sentence_to_id)

            for labels in self.flags:
                sentence_label = []
                for item in labels:
                    sentence_label.append(config.txt2label[item])
                sentence_labels.append(sentence_label)

            for word_list, labels in zip(sentences, sentence_labels):
                assert len(word_list) == len(labels)
                data.append((word_list, labels))
            return data
