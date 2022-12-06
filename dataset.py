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

    batch_len = len(sentence_list)
    # 这是这个批次里数据的最长长度
    max_len = max([len(s[0]) for s in sentence_list])
    max_true_word_mask = max([len(s[1]) for s in sentence_list])
    max_label = max(len(s) for s in labels_list)

    batch_data = torch.zeros(batch_len, max_len, dtype=torch.long)
    batch_labels = -1 * torch.ones(batch_len, max_label, dtype=torch.long)
    batch_true_word_mask = torch.zeros(batch_len, max_len, dtype=torch.long)
    _ = []
    __ = []
    for index, sen_list in enumerate(sentence_list):
        sen_list[0].extend([0] * (max_len - len(sen_list[0])))
        sen_list[1].extend([0] * (max_true_word_mask - len(sen_list[1])))
        __.append([0, ] + sen_list[1])
        _.append(sen_list[0])

    batch_data += torch.tensor(_)
    batch_true_word_mask += torch.tensor(__)

    for index, labels in enumerate(labels_list):
        batch_labels[index][:len(labels)] = torch.tensor(labels)

    if max_len > 512:
        batch_data = batch_data[:, :512]
        batch_true_word_mask = batch_true_word_mask[:, :512]
        batch_labels = batch_labels[:, :511]

    batch_data = batch_data.to(config.device)
    batch_labels = batch_labels.to(config.device)
    batch_true_word_mask = batch_true_word_mask.to(config.device)

    return [batch_data, batch_true_word_mask, batch_labels]


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
        sentences = self.data[index][0]
        flags = self.data[index][1]
        return [sentences, flags]

    def __len__(self):
        return len(self.lines)

    def process(self, mode: str):
        """
        :param mode: 要训练的模式
        :return: data 是一个二元组，第一个元素是 句子id 和句子掩码。第二个元素是单词对应的flag的集合
        """
        data = []
        data_dir = os.path.join(os.getcwd(), "data")

        with open(os.path.join(data_dir, f"{mode}.txt"), mode='r', encoding='utf-8') as f:
            list = f.readlines()
            for line in list:
                if len(line) <= 1:
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
            for sentence in self.lines:
                cls_sentence = ['[CLS]'] + [token for token in sentence]
                sentence_to_id = [self.tokenizer.convert_tokens_to_ids(word) for word in cls_sentence]
                sentences.append((copy.deepcopy(sentence_to_id), [1] * len(sentence)))
            for labels in self.flags:
                sentence_label = []
                for item in labels:
                    sentence_label.append(config.txt2label[item])
                sentence_labels.append(copy.deepcopy(sentence_label))

            for sentence, labels in zip(sentences, sentence_labels):
                data.append((sentence, labels))
            return data
