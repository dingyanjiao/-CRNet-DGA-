import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from typing import Generator,Union,List,NamedTuple,Optional,Tuple
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import os

class Example(NamedTuple):
	label : int  #是否为恶意地址
	char_ids : List[int] #每个字符在vocab中的id

class Vocab:
	def __init__(self):
		self.size = 1
		self.idx2char = ["[PAD]","[UNK]"]
		self.char2idx = {"[PAD]" : 0 , "[UNK]":1}

	def __add__(self, char):
		if char not in self.char2idx:
			self.idx2char.append(char)
			self.char2idx[char] = len(self.idx2char) - 1
			self.size += 1
		return self

	def __getitem__(self, item) -> Union[None,int,str]:
		if isinstance(item,str):
			if item not in self.char2idx : return 0
			return self.char2idx[item]
		if isinstance(item,int):
			return self.idx2char[item]

class Loginfo:
	def __init__(self, log_dir,logger_name):
		self.log_dir = log_dir
		self.logger_name = logger_name
		self.logger = self.init_logger(self.logger_name)

	def init_logger(self,logger_name):
		if logger_name not in Logger.manager.loggerDict:
			logger = logging.getLogger(logger_name)
			logger.setLevel(logging.DEBUG)
			handler = TimedRotatingFileHandler(filename = os.path.join(self.log_dir,"%s.txt" % logger_name), when='D', backupCount=7)
			datefmt = '%Y-%m-%d %H:%M:%S'
			format_str = '[%(asctime)s]: %(name)s [line:%(lineno)s] %(levelname)s  %(message)s'
			formatter = logging.Formatter(format_str, datefmt)
			handler.setFormatter(formatter)
			handler.setLevel(logging.INFO)
			logger.addHandler(handler)
			console = logging.StreamHandler()
			console.setLevel(logging.INFO)
			console.setFormatter(formatter)
			logger.addHandler(console)

		logger = logging.getLogger(logger_name)
		return logger


class DataLoader:

    def __init__(self, benign_data_path, malicious_data_path):
        max_domain_len = 67
        input_files = [malicious_data_path, benign_data_path]
        #self.data_processor = DataProcessor(benign_data_path, malicious_data_path)
        #self.examples = self.data_processor.preprocess()
        self.data_processor = DataPreprocessor(max_domain_len)
        self.examples = self.data_processor.create_examples_and_vocab(input_files)
        
		
        self.train_sets = []
        self.test_sets = []
        
        random.shuffle(self.examples)
        train_num = int(len(self.examples) * 7  / 10)  # 训练集数据
        self.train_sets = self.examples[: train_num]
        self.test_sets = self.examples[train_num:]
        print("数据集大小:",len(self.examples))
        print("训练集大小:",len(self.train_sets))
        print("验证集大小:",len(self.test_sets))
        
    def get_data(self):
        process_data = (self.train_sets, self.test_sets)
        torch.save(process_data,'/home/hanly/examples_2.pt')
        return self.train_sets, self.test_sets

  
class DataPreprocessor:
	def __init__(self, max_domain_len):
		self.max_domain_len=max_domain_len

	def create_examples_and_vocab(self, input_file_names : str):
		examples = []
		vocab = Vocab()
		max_domain_len = 0
		for filename in input_file_names:
			df = pd.read_csv(open(filename,encoding="utf-8"),dtype={'label':str})
			for idx in tqdm(range(len(df)),desc="转换文件%s中" % filename,total=len(df)):
				row = df.iloc[idx]#提取表格每一行
				domain_name = str(row['url']).replace(" ","").replace("\t","") #清洗域名，删除空格及制表符，其中row[0]代表表格第一列
				#is_black = not int(row['label'])#int(row['label']) 的值为0，则 not 0 会返回 True；如果 int(row['label']) 的值为非零整数，则 not 操作会返回 False
				is_black = int(row['label'] == '0')
				#更新最大域名长度
				if len(domain_name) > max_domain_len:
					max_domain_len = len(domain_name)

				char_ids = []
				#给字典添加字符
				for char in domain_name:
					vocab += char

				for char in domain_name:
					char_idx = vocab[char]
					char_ids.append(char_idx)

				assert len(char_ids) == len(domain_name)#验证 char_ids 的长度等于当前域名长度

				#补0
				if len(char_ids) <= self.max_domain_len:
					char_ids = char_ids + [0] * self.max_domain_len
				char_ids = char_ids[:self.max_domain_len]

				assert len(char_ids) == self.max_domain_len#验证 char_ids 的长度等于当前最大域名长度
				'''example = Example(
					char_ids = char_ids,
					label = is_black
				)'''
				examples.append((char_ids, is_black))
				
				log_dir = "/home/hanly/malicious_domain_name_detection/dingyj/DGA-Domain-Detection-master/log/"
				loginfo = Loginfo(log_dir,"DataPreprocess")
				
			
		loginfo.logger.info("vocab size : %d " % vocab.size)
		loginfo.logger.info("max domain len : %d " % max_domain_len)
		return examples

class DataProcessor:

    def __init__(self, benign_data_path, malicious_data_path, sequence_length=67):
        self.sequence_length = sequence_length
        self.vocab = self._build_vocab()
        self.char_vocab = CharVocab(self.vocab)
        
        benign_df = pd.read_csv(benign_data_path,encoding='utf-8', sep=',',dtype={'label':str})
        malicious_df = pd.read_csv(malicious_data_path,encoding='utf-8', sep=',',dtype={'label':str})
        
        benign=benign_df[benign_df['label']=='1']
        malicious=malicious_df[malicious_df['label']=='0']

        self.X = pd.concat([malicious, benign])

    def _build_vocab(self):
        vocab = {c:i+2 for i,c in enumerate('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-._~:/?#[]@!$&'()*+,;%=''')}
        vocab['[PAD]'] = 0
        vocab['[UNK]'] = 1
        print("vocab size :",len(vocab))
        return vocab

    def preprocess(self):
        examples = []
        for _, row in tqdm(self.X.iterrows(), total=len(self.X)):
            tokens = [self.char_vocab.ctoi(c) for c in row['url']][:self.sequence_length]
            tokens += [self.char_vocab.ctoi('[PAD]')] * (self.sequence_length - len(tokens)) 
            examples.append((tokens, int(row['label'] == '0')))
        return examples

class CharVocab:

    def __init__(self, vocab):
        self.vocab = vocab
  
    def ctoi(self, c):
        try:
            return self.vocab[c]
        except:
            return self.vocab['[UNK]']
		
if __name__ == "__main__":
	data = DataLoader('/home/hanly/malicious_domain_name_detection/data/dingyj/majestic_million.csv', '/home/hanly/malicious_domain_name_detection/data/dingyj/dga.csv')
	train_sets, test_sets = data.get_data()