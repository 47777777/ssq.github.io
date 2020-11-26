import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
'''
Embedding 是一个将离散变量转为连续向量表示的一个方式。

torch.nn.Embedding:
词嵌入在 pytorch 中非常简单，只需要调用 torch.nn.Embedding(m, n) 就可以了，
m 表示单词的总数目，n 表示词嵌入的维度，
其实词嵌入就相当于是一个大矩阵，矩阵的每一行表示一个单词。

Dropout就是在不同的训练过程中随机扔掉一部分神经元。
也就是让某个神经元的激活值以一定的概率p，让其停止工作

// 保留整数除运算 5//2=2
'''
class Bilstm(nn.Module):
    def __init__(self,vocab, embed_size, num_hiddens, num_layers,Dropout,label_size,seed_num):
        super(Bilstm, self).__init__()   #执行父类的构造函数，使得我们能够调用父类的属性。

        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)

        self.embedding = nn.Embedding(len(vocab),embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=False,
                               dropout=Dropout,
                               bidirectional=True)
        self.dropout=nn.Dropout(p=Dropout)
        self.decoder=nn.Linear(num_layers*num_hiddens, num_hiddens//2)
        self.sig = nn.ReLU()
        self.out = nn.Linear(num_hiddens//2 ,label_size)

    def forward(self,inputs,list1):
        # print(inputs)
        # print(inputs.shape)
        # print(list1)
        # print(list1.shape)
        embeddings = self.embedding(inputs.permute(1, 0))
        # print(list1)
        # print(embeddings.shape)
        embed_pad = pack_padded_sequence(embeddings, list1, batch_first=False, enforce_sorted=False)
        outputs, _ = self.encoder(embed_pad)  # output, (h, c)

        embed_pack, _ = pad_packed_sequence(outputs, batch_first=False)
        # print(embed_pack.shape)
        temp = embed_pack.permute(1,2,0)

        encoding = F.max_pool1d(temp, temp.size(2))

        temp = self.decoder(F.relu_(torch.squeeze(encoding)))
        outs = self.out(self.sig(temp))

        return outs




