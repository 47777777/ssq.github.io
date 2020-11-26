import torch
import time
import random
import preprocess
import pandas as pd
import torch.nn as nn
import evaluate.score as score
import models.bilstm as bilstm
import torch.utils.data as Data
import torchtext.vocab as pre_Vocab
# from sklearn.metrics import f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(20)
#数据读入
train = preprocess.data_loading("./data/cr.train.txt")
dev = preprocess.data_loading("./data/cr.dev.txt")
test = preprocess.data_loading("./data/cr.test.txt")

#词表创建
train_dev_test=train+dev+test
word_dict = preprocess.bulid_dict(train_dev_test)
vocab = preprocess.bulid_vocab(word_dict)
# print(vocab)
batch_size = 128
print("-"*20+"词表创建完成"+"-"*20)

#训练集合的组装
train_word_batch , train_lable_batch ,train_seq_len_batch = preprocess.bulid_batch(train,vocab)
train_dataset = Data.TensorDataset(train_word_batch , train_lable_batch ,train_seq_len_batch)
train_loader = Data.DataLoader(train_dataset,batch_size,True)
#验证集合组装
dev_word_batch , dev_lable_batch ,dev_seq_len_batch = preprocess.bulid_batch(dev,vocab)
dev_dataset = Data.TensorDataset(dev_word_batch , dev_lable_batch,dev_seq_len_batch)
dev_loader = Data.DataLoader(dev_dataset,batch_size)
#测试集合组装
test_word_batch , test_lable_batch ,test_seq_len_batch = preprocess.bulid_batch(test,vocab)
test_dataset = Data.TensorDataset(test_word_batch , test_lable_batch,test_seq_len_batch)
test_loader = Data.DataLoader(test_dataset,batch_size)


# bilstm参数
embed_size = 200
num_hiddens = 200
num_layers = 2
dropout = 0.1
label_size = 2
seed_num = 20
model = bilstm.Bilstm(vocab, embed_size, num_hiddens, num_layers,dropout,label_size,seed_num)
#训练参数
num_epochs = 50
lr = 0.01


# 验证集合函数
def pred(dev_loader,model,device=None):
    if device is None:
        device = list(model.parameters())[0].device
    label = [] #存放预测的标签
    label_true = []   #存放真实的标签
    model = model.to(device)
    with torch.no_grad():  #不生成计算图，显著减少显存占用
        model.eval()
        for X, Y,z in dev_loader:
            label.extend(torch.argmax(model(X.to(device),z), dim=1).cpu().numpy().tolist())
            label_true.extend(Y.numpy().tolist())
        model.train()
    #计算F1 score
    return score.f1(label, label_true, classifications=2)


def train(train_loader,dev_loader,model,loss,optimizer,scheduler,device,num_epochs):
    model = model.to(device)
    print("开始训练",device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y,z in train_loader:
            X = X.to(device)
            y = y.to(device)
            # # print(X)
            # print(X.shape)
            # # print("-" * 40)
            # # print(z)
            # print(z.shape)
            y_hat = model(X,z)
            l = loss(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            # 梯度下降
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            dev_f1 = pred(dev_loader, model, device)  # 计算验证集f1分数
            scheduler.step(dev_f1)
            if (dev_f1 > 0.781):
                print(pred(test_loader, model, device))
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, dev_f1score %.3f,time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,dev_f1, time.time() - start))


# 预训练词向量使用
cache_dir = r'F:\glove.6B'
glove_vocab = pre_Vocab.GloVe(name='6B', dim=200, cache=cache_dir)

def load_pretrained_embedding(words, pretrained_vocab):
    '''
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
    @return:
        embed: 加载到的词向量
    '''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为len*100维度
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]# 将每个词语用训练的语言模型理解，pad,unkown
        except KeyError:
            # if i!=1:
            #     idx = pretrained_vocab.stoi['<unk>']
            #     embed[i, :] = pretrained_vocab.vectors[idx]
            # else:
            #     embed[i,:]=embed[i,:]
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    # print(embed.shape),在词典中寻找相匹配的词向量
    return embed


model.embedding.weight.data.copy_(load_pretrained_embedding(vocab, glove_vocab))
model.embedding.weight.requires_grad = True # 直接加载预训练好的, 所以不需要更新它


optimizer = torch.optim.Adam(model.parameters(),lr=lr)
# 指数调整
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.5,patience=10,verbose=True,eps=0.000005)#f1_score is :0.7605179083076073
loss = torch.nn.CrossEntropyLoss()# softmax,交叉熵
train(train_loader,dev_loader,model,loss, optimizer,scheduler, device, num_epochs)


# 测试
label = []
label_true = []
model = model.to(device)
for X,Y,z in test_loader:
  label.extend(torch.argmax(model(X.to(device),z),dim=1).cpu().numpy().tolist())
  label_true.extend(Y.numpy().tolist())
print("accuracy is :{}".format(score.accuracy(label,label_true)))
print('f1_score is :{}'.format(score.f1(label,label_true,2)))


# 没加预训练模块
# --------------------词表创建完成--------------------
# 开始训练 cuda
# epoch 1, loss 0.6110, train acc 0.652, dev_f1score 0.708,time 3.6 sec
# Epoch    39: reducing learning rate of group 0 to 5.0000e-03.
# epoch 2, loss 0.1928, train acc 0.819, dev_f1score 0.725,time 3.2 sec
# Epoch    54: reducing learning rate of group 0 to 2.5000e-03.
# 0.7959736235754422
# 0.7796762304295569
# epoch 3, loss 0.0487, train acc 0.952, dev_f1score 0.777,time 3.3 sec
# Epoch    78: reducing learning rate of group 0 to 1.2500e-03.
# Epoch    89: reducing learning rate of group 0 to 6.2500e-04.
# epoch 4, loss 0.0121, train acc 0.988, dev_f1score 0.771,time 3.2 sec
# Epoch   100: reducing learning rate of group 0 to 3.1250e-04.
# Epoch   111: reducing learning rate of group 0 to 1.5625e-04.
# epoch 5, loss 0.0054, train acc 0.994, dev_f1score 0.771,time 3.3 sec
# Epoch   122: reducing learning rate of group 0 to 7.8125e-05.
# Epoch   133: reducing learning rate of group 0 to 3.9063e-05.
# Epoch   144: reducing learning rate of group 0 to 1.9531e-05.
# epoch 6, loss 0.0040, train acc 0.995, dev_f1score 0.775,time 3.3 sec
# Epoch   155: reducing learning rate of group 0 to 9.7656e-06.
# epoch 7, loss 0.0032, train acc 0.995, dev_f1score 0.775,time 3.3 sec
# epoch 8, loss 0.0029, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 9, loss 0.0025, train acc 0.995, dev_f1score 0.775,time 3.3 sec
# epoch 10, loss 0.0022, train acc 0.995, dev_f1score 0.775,time 3.4 sec
# epoch 11, loss 0.0020, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 12, loss 0.0018, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 13, loss 0.0016, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 14, loss 0.0015, train acc 0.995, dev_f1score 0.775,time 3.3 sec
# epoch 15, loss 0.0014, train acc 0.995, dev_f1score 0.775,time 3.3 sec
# epoch 16, loss 0.0013, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 17, loss 0.0012, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 18, loss 0.0011, train acc 0.995, dev_f1score 0.775,time 3.2 sec
# epoch 19, loss 0.0011, train acc 0.995, dev_f1score 0.775,time 3.4 sec
# epoch 20, loss 0.0010, train acc 0.995, dev_f1score 0.775,time 3.4 sec
# epoch 21, loss 0.0010, train acc 0.996, dev_f1score 0.775,time 3.7 sec
# epoch 22, loss 0.0009, train acc 0.996, dev_f1score 0.775,time 3.3 sec
# epoch 23, loss 0.0008, train acc 0.996, dev_f1score 0.775,time 3.2 sec
# epoch 24, loss 0.0008, train acc 0.996, dev_f1score 0.775,time 3.2 sec
# epoch 25, loss 0.0008, train acc 0.996, dev_f1score 0.775,time 3.2 sec
# epoch 26, loss 0.0007, train acc 0.996, dev_f1score 0.775,time 3.2 sec
# epoch 27, loss 0.0007, train acc 0.996, dev_f1score 0.775,time 3.3 sec
# epoch 28, loss 0.0007, train acc 0.996, dev_f1score 0.775,time 3.2 sec
# epoch 29, loss 0.0006, train acc 0.996, dev_f1score 0.777,time 3.3 sec
# epoch 30, loss 0.0006, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 31, loss 0.0006, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 32, loss 0.0005, train acc 0.996, dev_f1score 0.777,time 3.3 sec
# epoch 33, loss 0.0005, train acc 0.996, dev_f1score 0.777,time 3.3 sec
# epoch 34, loss 0.0005, train acc 0.996, dev_f1score 0.777,time 3.3 sec
# epoch 35, loss 0.0005, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 36, loss 0.0005, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 37, loss 0.0004, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 38, loss 0.0004, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 39, loss 0.0004, train acc 0.996, dev_f1score 0.777,time 3.2 sec
# epoch 40, loss 0.0004, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 41, loss 0.0004, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 42, loss 0.0004, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 43, loss 0.0004, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 44, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 45, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 46, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 47, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 48, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 49, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# epoch 50, loss 0.0003, train acc 0.997, dev_f1score 0.777,time 3.2 sec
# accuracy is :0.8010752688172043
# f1_score is :0.7714152350580268




# 加了预训练模块
# --------------------词表创建完成--------------------
# There are 443 oov words.
# 开始训练 cuda
# epoch 1, loss 0.6545, train acc 0.607, dev_f1score 0.640,time 6.1 sec
# epoch 2, loss 0.2343, train acc 0.770, dev_f1score 0.726,time 3.2 sec
# 0.7469797066594548
# 0.7483876554249782
# 0.7519296160499537
# 0.7398600895300621
# 0.7375304791307709
# Epoch    68: reducing learning rate of group 0 to 5.0000e-03.
# epoch 3, loss 0.0772, train acc 0.907, dev_f1score 0.751,time 3.5 sec
# Epoch    79: reducing learning rate of group 0 to 2.5000e-03.
# Epoch    90: reducing learning rate of group 0 to 1.2500e-03.
# epoch 4, loss 0.0261, train acc 0.966, dev_f1score 0.748,time 3.2 sec
# Epoch   101: reducing learning rate of group 0 to 6.2500e-04.
# Epoch   112: reducing learning rate of group 0 to 3.1250e-04.
# epoch 5, loss 0.0119, train acc 0.981, dev_f1score 0.742,time 3.2 sec
# Epoch   123: reducing learning rate of group 0 to 1.5625e-04.
# Epoch   134: reducing learning rate of group 0 to 7.8125e-05.
# epoch 6, loss 0.0084, train acc 0.986, dev_f1score 0.749,time 3.2 sec
# Epoch   145: reducing learning rate of group 0 to 3.9063e-05.
# Epoch   156: reducing learning rate of group 0 to 1.9531e-05.
# Epoch   167: reducing learning rate of group 0 to 9.7656e-06.
# epoch 7, loss 0.0070, train acc 0.987, dev_f1score 0.749,time 3.2 sec
# epoch 8, loss 0.0062, train acc 0.987, dev_f1score 0.749,time 3.2 sec
# epoch 9, loss 0.0054, train acc 0.987, dev_f1score 0.749,time 3.2 sec
# epoch 10, loss 0.0049, train acc 0.987, dev_f1score 0.749,time 3.2 sec
# epoch 11, loss 0.0045, train acc 0.987, dev_f1score 0.749,time 3.2 sec
# epoch 12, loss 0.0040, train acc 0.988, dev_f1score 0.749,time 3.2 sec
# epoch 13, loss 0.0037, train acc 0.988, dev_f1score 0.749,time 3.2 sec
# epoch 14, loss 0.0034, train acc 0.988, dev_f1score 0.749,time 3.3 sec
# epoch 15, loss 0.0033, train acc 0.988, dev_f1score 0.749,time 3.3 sec
# epoch 16, loss 0.0030, train acc 0.987, dev_f1score 0.749,time 3.3 sec
# epoch 17, loss 0.0028, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 18, loss 0.0026, train acc 0.988, dev_f1score 0.749,time 3.2 sec
# epoch 19, loss 0.0025, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 20, loss 0.0024, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 21, loss 0.0022, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 22, loss 0.0021, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 23, loss 0.0021, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 24, loss 0.0019, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 25, loss 0.0019, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 26, loss 0.0017, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 27, loss 0.0017, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 28, loss 0.0016, train acc 0.988, dev_f1score 0.746,time 3.2 sec
# epoch 29, loss 0.0015, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 30, loss 0.0015, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 31, loss 0.0015, train acc 0.990, dev_f1score 0.746,time 3.2 sec
# epoch 32, loss 0.0014, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 33, loss 0.0014, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 34, loss 0.0013, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 35, loss 0.0013, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 36, loss 0.0012, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 37, loss 0.0012, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 38, loss 0.0012, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 39, loss 0.0011, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 40, loss 0.0011, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 41, loss 0.0010, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 42, loss 0.0010, train acc 0.989, dev_f1score 0.746,time 3.2 sec
# epoch 43, loss 0.0010, train acc 0.989, dev_f1score 0.743,time 3.2 sec
# epoch 44, loss 0.0010, train acc 0.989, dev_f1score 0.743,time 3.2 sec
# epoch 45, loss 0.0009, train acc 0.989, dev_f1score 0.743,time 3.2 sec
# epoch 46, loss 0.0009, train acc 0.989, dev_f1score 0.743,time 3.2 sec
# epoch 47, loss 0.0009, train acc 0.989, dev_f1score 0.743,time 3.2 sec
# epoch 48, loss 0.0009, train acc 0.989, dev_f1score 0.743,time 3.2 sec
# epoch 49, loss 0.0008, train acc 0.990, dev_f1score 0.743,time 3.2 sec
# epoch 50, loss 0.0008, train acc 0.990, dev_f1score 0.743,time 3.2 sec
# accuracy is :0.7876344086021505
# f1_score is :0.7676892535068931

# batch_size=64
# accuracy is :0.7903225806451613
# f1_score is :0.7685902453482525