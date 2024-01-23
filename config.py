import os
import platform

import torch

sys_platform = platform.platform().lower()

if "windows" in sys_platform:
    # 在 windows 下，将这里的路径换成你的 bert-base-chinese 路径
    bert_path = r"C:/Users/hengy/bert-base-chinese"
else:
    # 在其他系统下，以linux下为例， bert-base-chinese 放在了用户目录下
    home_path = os.environ['HOME']
    bert_path = os.path.join(home_path, "bert-base-chinese")
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-5
ner_classes_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]", "[PAD]"]
batch_size = 8
save_path = "contractNerEntity.pth"
max_length = 128
last_state_dim = 768
dropout = 0.5
epoches = 20
bert_path = bert_path
device = device
txt2label = {text: index for index, text in enumerate(ner_classes_list)}
label2txt = {index: text for index, text in enumerate(ner_classes_list)}
