from torch.utils.data import DataLoader
import torch

from Model import BertNer
import config
from tqdm import tqdm


def ner_test(model: BertNer, data_loader: DataLoader, epoch: int):
    model.to(device=config.device)
    model.eval()
    with torch.no_grad():
        total_acc = 0
        total_count = 0
        for batch_data in tqdm(data_loader):
            batch_sentence_id, batch_sentence_mask, batch_label = batch_data
            batch_masks = batch_sentence_id.gt(0)
            logits = model(batch_sentence_id=batch_sentence_id, batch_mask=batch_masks,
                           batch_sentence_mask=batch_sentence_mask, batch_labels=None)[0]
            true_tags = []
            bacth_output = model.crf.decode(logits, mask=batch_label.gt(-1))
            batch_tags = batch_label.to('cpu').numpy()
            true_tags.extend([[idx for idx in indices if idx > -1] for indices in batch_tags])
            for pre, real in zip(bacth_output, true_tags):
                assert len(pre) == len(real), "pre长{},real长{}".format(len(pre), len(real))
                acc = (torch.tensor(pre) == torch.tensor(real)).sum().item()
                total_acc += acc
                total_count += len(pre)
    print("第{}轮的精度是{}".format(epoch, total_acc / total_count))
    return total_acc / total_count


def ner_train(model: BertNer, train_data_loader, epoch: int, learning_rate: float = 1e-5):
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss_train = 0
    for batch_data in tqdm(train_data_loader):
        batch_sentence_id, batch_sentence_mask, batch_label = batch_data
        batch_masks = batch_sentence_id.gt(0)
        loss = \
            model(batch_sentence_id=batch_sentence_id, batch_mask=batch_masks, batch_sentence_mask=batch_sentence_mask,
                  batch_labels=batch_label)[0]
        total_loss_train += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = float(total_loss_train) / len(train_data_loader)
    print("Epoch:{},train loss ：{}".format(epoch, train_loss))
