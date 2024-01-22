import copy

import torch.nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
import config
import os


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
        input_id = [101] + x + [102]  # 添加 [CLS]和[SEP]
        token_type_id = [0] * max_len
        attention = [1] * len(input_id) + [0] * (max_len - len(input_id))
        label = [8] + y + [9] + [10] * (max_len - len(input_id))  # 增加标签
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
    # batch_len = len(sentence_list)
    # # 这是这个批次里数据的最长长度
    # max_len = max([len(s[0]) for s in sentence_list])
    # max_true_word_mask = max([len(s[1]) for s in sentence_list])
    # max_label = max(len(s) for s in labels_list)
    #
    # batch_data = torch.zeros(batch_len, max_len, dtype=torch.long)
    # batch_labels = -1 * torch.ones(batch_len, max_label, dtype=torch.long)
    # batch_true_word_mask = torch.zeros(batch_len, max_len, dtype=torch.long)
    # _ = []
    # __ = []
    # for index, sen_list in enumerate(sentence_list):
    #     sen_list[0].extend([0] * (max_len - len(sen_list[0])))
    #     sen_list[1].extend([0] * (max_true_word_mask - len(sen_list[1])))
    #     __.append([0, ] + sen_list[1])
    #     _.append(sen_list[0])
    #
    # batch_data += torch.tensor(_)
    # batch_true_word_mask += torch.tensor(__)
    #
    # for index, labels in enumerate(labels_list):
    #     batch_labels[index][:len(labels)] = torch.tensor(labels)
    #
    # if max_len > 512:
    #     batch_data = batch_data[:, :512]
    #     batch_true_word_mask = batch_true_word_mask[:, :512]
    #     batch_labels = batch_labels[:, :511]
    #
    # batch_data = batch_data.to(config.device)
    # batch_labels = batch_labels.to(config.device)
    # batch_true_word_mask = batch_true_word_mask.to(config.device)
    #
    # return [batch_data, batch_true_word_mask, batch_labels]


class NerDataSet(Dataset):
    def __init__(self, mode: str):
        self.max_length = config.max_length
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
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
                # cls_sentence = ['[CLS]'] + word_list + ['[SEP]']
                # 原来的汉字被转化为了token_id
                sentence_to_id = [self.tokenizer.convert_tokens_to_ids(word) for word in word_list]
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
