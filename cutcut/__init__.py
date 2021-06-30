#encoding=utf8
import functools
import json
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pdb
import random
import sys
import time
import warnings

import bert
import numpy as np
import pkg_resources
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import (StockBertConfig, load_stock_weights,
                         map_stock_config_to_params)

warnings.filterwarnings('ignore')

get_module_datapath = lambda *res: pkg_resources.resource_filename(__name__, os.path.join(*res))

def input_fn(str_list, batch_size):
    #预处理
    vocab_file_path = get_module_datapath('data/vocab.txt')
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file = vocab_file_path)
    batch_size = batch_size
    max_seq_len = 50
    def parse_fn(line):
        tmp_list = line.strip().split('\t')
        # text = tmp_list[0].strip()
        text = tmp_list[0].strip().replace(' ', '')
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        words = tokenizer.convert_tokens_to_ids(tokens)

        nwords = len(words)
        words = words[:min(nwords, max_seq_len)]
        words = words + [0] * (max_seq_len - nwords)

        return (words, 0)

    def generator_fn(input_list):
        for line in input_list:
            yield parse_fn(line)
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    shapes = ([None],())
    types = (tf.int64, tf.int32)
    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, str_list),output_shapes=shapes, output_types=types)
    dataset = dataset.batch(batch_size)
    return dataset

class Cutcut(object):
    '''
    基于Albert模型的通用分词器，使用标准数据集cityU、MSRhe PKU进行训练。
    包含方法: lcut(str), cut(str)和batch_lcut(str_list)。
    '''    
    def __init__(self):
        self.wordIndexMap = []
        self.tagIndexMap = {}
        self.model_path = get_module_datapath('savedModel')
        self.initialized = False
    
    def initialize(self):
        if self.initialized:
            return
        with open(get_module_datapath('data/vocab.txt'), 'r', encoding='utf8') as vocab_file:
            for line in vocab_file:
                self.wordIndexMap.append(line.strip())
        with open(get_module_datapath('data/tag.txt'), 'r', encoding='utf8') as tag_file:
            for idx, line in enumerate(tag_file):
                self.tagIndexMap[idx] = line.strip()
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.initialized = True

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def lcut(self, str):
        '''
        @Description: 将输入字符串str切分为若干个单词或字符
        @Args: 输入字符串
        @Returns: 切分后的单词列表
        '''
        self.check_initialized()
        str_list = [str]
        batch_size = 1
        res = []
        dataset = input_fn(str_list, batch_size)
        map_func = lambda x:self.wordIndexMap[x]
        vfunc = np.vectorize(map_func)
        for (words, _) in dataset:
            results = self.model(words)
            ner_logits = results['ner_logits']
            pred_tags = np.argmax(ner_logits, 2)
            # pdb.set_trace()
            src_texts = vfunc(words)
            for text, tag in zip(src_texts, pred_tags):
                tmp_str = ''
                sep_idx = list(text).index('[SEP]')
                for idx in range(1, sep_idx):
                    if text[idx]:
                        t = tag[idx]
                        if t == 0 or t == 1 or t == 2:
                            continue
                        else:
                            if t == 3 or t == 4:
                                tmp_str += text[idx]
                            elif t == 5:
                                tmp_str += text[idx] + '/'
                            elif t == 6:
                                tmp_str += '/' + text[idx] + '/'
                tmp_res = tmp_str.replace('//', '/').split('/')
                tmp_res = [token for token in tmp_res if token != '[UNK]' and token != '']
                res.append(tmp_res)
        return res[0]

    def cut(self, str):
        '''
        @Description: 将输入字符串str切分为若干个单词或字符
        @Args: 输入字符串
        @Returns: 切分后的单词序列迭代器
        '''        
        self.check_initialized()
        return iter(self.lcut(str))
    
    def batch_lcut(self, str_list):
        '''
        @Description: 将输入的多个字符串切分为对应的若干个单词或字符
        @Args: 输入字符串列表
        @Returns: 包含切分后的单词列表的列表
        '''        
        self.check_initialized()
        batch_size = 256
        res = []
        dataset = input_fn(str_list, batch_size)
        map_func = lambda x:self.wordIndexMap[x]
        vfunc = np.vectorize(map_func)
        for (words, _) in dataset:
            results = self.model(words)
            ner_logits = results['ner_logits']
            pred_tags = np.argmax(ner_logits, 2)
            src_texts = vfunc(words)
            for text, tag in zip(src_texts, pred_tags):
                tmp_str = ''
                sep_idx = list(text).index('[SEP]')
                for idx in range(1, sep_idx):
                    if text[idx]:
                        t = tag[idx]
                        if t == 0 or t == 1 or t == 2:
                            continue
                        else:
                            if t == 3 or t == 4:
                                tmp_str += text[idx]
                            elif t == 5:
                                tmp_str += text[idx] + '/'
                            elif t == 6:
                                tmp_str += '/' + text[idx] + '/'
                tmp_res = tmp_str.replace('//', '/').split('/')
                tmp_res = [token for token in tmp_res if token != '[UNK]' and token != '']
                res.append(tmp_res)
        return res
    _lcut = lcut

cutcut = Cutcut()

lcut = cutcut.lcut
cut = cutcut.cut
batch_lcut = cutcut.batch_lcut

def _lcut(s):
    return cutcut._lcut(s)
