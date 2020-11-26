import os
import re
import torch
import collections

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`|]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() #去除首尾空格


def data_loading(fpath):
    #数据加载和分词
    sents=[]
    with open(fpath,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            # sent = clean_str(' '.join(line[2:]))
            sent=line[2:]
            seq_len = len(sent)
            lable=int(line[0])
            sents.append((sent,lable,seq_len))
    # print(sents)
    return sents



def bulid_dict(sents):
    '''

    :param  sents:  [(['i', 'like', 'you', 'about'], 1,4),(['no', 'way'], 0,2)]
    :return: OrderedDict([('i', 1), ('like', 1), ('it', 1), ('no', 1), ('way', 1)])
    '''
    word_dict = collections.OrderedDict()
    for k in sents:
        words = k[0]
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1

    return word_dict


def bulid_vocab(word_dict):
    '''

    :param word_dict:
    :return: OrderedDict([('i', 1), ('like', 2), ('it', 3), ('no', 4), ('way', 5), ('unk',6),('padding',7)])
    '''
    word_dict['unk']=100
    word_dict['padding']=100
    word_vocab=collections.OrderedDict()
    min=0
    word_idx=1
    for key in word_dict:
        if word_dict[key]>min:
            if word_dict[key] not in word_vocab:
               word_vocab[key]=word_idx
               word_idx=word_idx+1

    # print('样例（单词->索引）:')
    # print(word_vocab)
    # print('词表大小:',len(word_vocab))
    return word_vocab

def get_indexs(words,word_vocab):
    '''

    :param sents:  ['i', 'like', 'you', 'about']
    :param word_vocab:
    :return:   [1,2,3,4]

    '''
    word_indexs=[]
    for word in words:
        if word in word_vocab:
            index = word_vocab[word]
        else:
            index = word_vocab['unk']
        word_indexs.append(index)
    # print(word_indexs)
    return word_indexs


def bulid_batch(sents,word_vocab):
    '''

    :param sents:
    :param word_vocab:
    :return:
    tensor([[  1,   2,   3,  ...,   0,   0,   0],
        [  6,  23,  24,  ...,   0,   0,   0],
        [ 42,  43,  24,  ...,  66,  67,  12],
        ...,
        [128, 514,   6,  ...,   0,   0,   0],
        [ 28,   6, 789,  ...,   0,   0,   0],
        [  1,  46,  50,  ...,   0,   0,   0]])
    tensor([1, 1, 0,  ..., 0, 1, 1])
    tensor([26, 31, 63,  ...,  8,  7, 16])


    '''
    sentence_size = 50
    word_batch=[]
    lable_batch=[]
    seq_len_batch=[]
    word_indexs=[]
    for t in sents:
        words = t[0]
        lable = t[1]
        seq_len = t[2]
        word_indexs = get_indexs(words,word_vocab)
        if len(word_indexs) > sentence_size:
            word_indexs=word_indexs[:50]
        elif len(word_indexs) < sentence_size:
            for t in range(len(word_indexs),50):
                word_indexs.append(0)
        word_batch.append(word_indexs)
        lable_batch.append(lable)
        if seq_len>50:
            seq_len_batch.append(50)
        else:
            seq_len_batch.append(seq_len)

    word_batch , lable_batch ,seq_len_batch = torch.LongTensor(word_batch),torch.LongTensor(lable_batch),torch.LongTensor(seq_len_batch)
    # print(word_batch.shape)
    # print(lable_batch.shape)
    # print(seq_len_batch.shape)
    return word_batch,lable_batch,seq_len_batch


#
# sents=data_loading("F:/nlpstart/data/cr.train.txt")
# word_dict= bulid_dict(sents)
# word_vocab = bulid_vocab(word_dict)
# word_batch , lable_batch , seq_len_batch = bulid_batch(sents,word_vocab)
