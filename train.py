import random

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from Model import BertNer
from dataset import NerDataSet, collate_fn


@torch.no_grad()
def ner_test(model: BertNer, data_loader: DataLoader):
    model.eval()
    true_tags = []
    pred_tags = []
    for bert_input, batch_label in tqdm(data_loader, position=0, leave=True):
        logits = model(bert_input, batch_label)[1]

        batch_output = model.crf.decode(logits, mask=bert_input['attention_mask'].gt(0))
        batch_tags = batch_label.detach().cpu().tolist()

        for pred, true in zip(batch_output, batch_tags):
            pred_tags.extend(pred)
            true_tags.extend(true[:len(pred)])
    p = precision_score(true_tags, pred_tags, average='micro')
    r = recall_score(true_tags, pred_tags, average='micro')
    f1 = f1_score(true_tags, pred_tags, average='micro')
    logger.info(f'precision_score: {p}')
    logger.info(f'recall_score: {r}')
    logger.info(f'f1_score: {f1}')
    return f1_score(true_tags, pred_tags, average='micro')


def ner_train(model: BertNer, train_data_loader, dev_data_loader, epoches: int, learning_rate: float = 1e-5):
    val_accuracy = 0.0
    start_stop = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss_train = 0
    for epoch in range(epoches):
        for index, data in enumerate(tqdm(train_data_loader, position=0, leave=True), start=1):
            bert_input, batch_label = data
            step = epoch * len(train_data_loader) + index
            loss = model(bert_input, batch_label)[0]
            total_loss_train += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                logger.info(f"平均误差{total_loss_train / 200}")
                total_loss_train = 0.0
                f1 = ner_test(model=model, data_loader=dev_data_loader)
                model.train()
                if f1 > val_accuracy:
                    start_stop = 0
                    torch.save(model.state_dict(), f"contractNerEntity.pth")
                    logger.info("模型已保存。")
                    val_accuracy = f1
                else:
                    start_stop += 1
                    if start_stop == 50:
                        logger.info(f"结束，效果最好的是{val_accuracy}")
                        return


def set_seed(seed_num=114514):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


if __name__ == '__main__':
    set_seed()
    model = BertNer().to(config.device)
    train_data_set = NerDataSet("train")
    dev_data_set = NerDataSet("dev")
    train_data_loader = DataLoader(dataset=train_data_set, shuffle=True, batch_size=config.batch_size,
                                   collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_data_set, shuffle=True, batch_size=config.batch_size,
                                 collate_fn=collate_fn)
    ner_train(model=model, train_data_loader=train_data_loader, dev_data_loader=dev_data_loader, epoches=config.epoches)
