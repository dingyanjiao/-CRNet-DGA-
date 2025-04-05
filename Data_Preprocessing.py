import pandas as pd
from tqdm import tqdm
import random
import torch
from typing import Union
import datetime
from loguru import logger

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

class DataLoader:

    def __init__(self, benign_data_path, malicious_data_path):
        max_domain_len = 67
        input_files = [malicious_data_path, benign_data_path]
        self.data_processor = DataPreprocessor(max_domain_len)
        self.examples = self.data_processor.create_examples_and_vocab(input_files)
        
        self.train_sets = []
        self.test_sets = []
        
        random.shuffle(self.examples)
        train_num = int(len(self.examples) * 7  / 10)  # 训练集数据
        self.train_sets = self.examples[: train_num]
        self.test_sets = self.examples[train_num:]
        logger.info("训练集大小为 : %d " % len(self.train_sets))
        logger.info("测试集大小为 : %d " % len(self.test_sets))
        
    def get_data(self):
        process_data = (self.train_sets, self.test_sets)
        #torch.save(process_data,'/home/hanly/examples_1.pt')
        return self.train_sets, self.test_sets

  
class DataPreprocessor:
	def __init__(self, max_domain_len):
		self.max_domain_len=max_domain_len

	def create_examples_and_vocab(self, input_file_names : str):
		examples = []
		vocab = Vocab()
		max_domain_len = 0
		for filename in input_file_names:
			df = pd.read_csv(open(filename),dtype={'label':str},encoding="utf-8")
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
				examples.append((char_ids, is_black))

		current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		log_file = f'/home/dingyj/malicious_domain_name_detection/dingyj/pytorch_malicious_url/log/dataprocessing_{current_time}.log'
		logger.add(log_file, level="INFO", encoding="utf-8")	
		logger.info("构建词典大小为 : %d " % vocab.size)
		logger.info("最长域名长度为 : %d " % max_domain_len)
		logger.info("数据集大小为 : %d " % len(examples))

		return examples

		
if __name__ == "__main__":
	data = DataLoader('/home/dingyj/malicious_domain_name_detection/data/dingyj/majestic_million.csv', '/home/dingyj/malicious_domain_name_detection/data/dingyj/dga.csv')
	train_sets, test_sets = data.get_data()