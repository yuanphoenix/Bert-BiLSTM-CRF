# Bert-BiLSTM-CRF
中文的 Bert+BiLSTM+CRF 命名实体识别任务

数据来自于人民日报标注的数据集，具体来源其实不是很清楚，我之前在 github 上找的，接下来主要是精简一下 dataset 的代码，已经补充一些注释。
我认为这个模型是正确的，包括一些细节。
____
这个模型的准确率还是可以的。比我之前从网上找的效果都要好，果然最好的东西还是自己造出来的。
____
训练的时候很吓人，有时候占用的显存特别大，
模型参考了 [https://github.com/CLUEbenchmark/CLUENER2020](https://github.com/hemingkx/CLUENER2020) 代码，特别是读取数据的部分。
