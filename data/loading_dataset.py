from datasets import load_dataset
import pandas as pd
import torch
import ast
import numpy




"""
krasaee/nietzsche dataset (for srs pretraining)
https://huggingface.co/datasets/krasaee/nietzsche
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python load_dataset.py
Will save shards to the local directory "nietzsche_shards".
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets



#downlaod and cache dataset
def download_and_cache_dataset(path:str,data_split="train",text='text')-> list:
    raw_data = load_dataset(path)
    raw_data_split  = raw_data[data_split]
    raw_texts = raw_data_split[text]
    print("raw text len: ", len(raw_texts)) ##9930147
    return raw_texts


#split into training, validation & testing datasets
def split_dataset(dataset:list, train_split:float,val_split:float, test_split:float) -> dict:
    train_chunk = int(train_split * len(dataset))
    val_chunk = int(val_split * len(dataset))
    test_chunk = int(test_split * len(dataset))

    train_set = dataset[0:train_chunk]
    val_set = dataset[train_chunk:train_chunk+val_chunk]
    test_set = dataset[train_chunk+val_chunk:]
    datasets  = {
        "train_set":train_set ,
        "val_set": val_set,
        "test_set": test_set,
    }
    #print(f"train set: {len(train_set)}, val set: {len(val_set)}, test set {len(test_set)}")
    return datasets


#save raw data as shards
def save_data_shards(file_path:str, texts:str, tokenize=False)-> None:
    data = ''
    data = tokenize(texts)  if tokenize else texts
    np.save(file_path, data)
    print(f"INFO: {file_path} written ! (size {len(data)})")

#save tokenized datasets
def save_tokenized_datasets(datasets:dict, enc:tiktoken.core.Encoding,dir:str)->None:
    for key, value in datasets.items():
        tokenized_dataset = tokenize(enc,value)
        dataset_path = get_dataset_path(dir, key)
        #print("dataset_path", dataset_path)
        save_data_shards(dataset_path,tokenized_dataset)


#tokenize texts
def tokenize(enc:tiktoken.core.Encoding, data:str)->numpy.ndarray:
    # tokenizes a single document and returns a numpy array of uint16 tokens
    data = ''.join(data) #convert into str type before tokenizaton
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(data)) #encode_ordinary converts strings into tokens(integers)
    tokens_np = np.array(tokens) #by default int64 dtype
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "insure that all the token IDs in tokens_np can safely be stored in a uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16) #save as unit unit16 dtype
    return tokens_np_uint16


#get tokensized dataset file path
def get_dataset_path(dir:str,dataset:str)->str:
    return f"{dir}_{dataset}_{0:06d}.npy"
    #return f"nietzsche_shards/nietzsche_{dataset}_{0:06d}.npy"

#load tokenized dataset
def load_tokens(filename:str)->torch.Tensor:
    tokenized_data = np.load(filename) #dtype unit16
    tokenized_data = tokenized_data.astype(np.int32)  #convert to int32 dtype
    tokenized_data = torch.tensor(tokenized_data, dtype=torch.long) #convert to long dtype
    return tokenized_data

#load tokenized datasets
def load_tokenized_datatsets(dir:str, datasets:dict)->dict:
    tokenized_datasets = {}
    for key, _ in datasets.items():
        tokenized_datasets[key] = load_tokens(get_dataset_path(dir, key))
        print(f'INFO(Loading Dataset): {key} of ({len(tokenized_datasets[key])}) loaded !')
    return tokenized_datasets


#DataLoaderLite for multiple processes GPUs
class DataLoaderLite:
    def __init__(self, B:int, T:int, process_rank:int, num_processes:int,tokenized_dataset:torch.Tensor):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokens = tokenized_dataset #load_tokens(file_path)
        self.tokenized_dataset = tokenized_dataset
        print(f"INFO(Dataset split): loaded {len(self.tokens)} tokens")  #2288478 tokens
        # starting point for each process(GPU)
        self.current_position = self.B * self.T * self.process_rank #first process starts at pos 0

    #next batch for each GPU process

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.tokenized_dataset #load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes #each process advances by the entire chunk to get next batch
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes+ 1) > len(self.tokens):
            print("Reached the end of the datast, reseting index...")
            self.current_position = self.B * self.T * self.process_rank
        return x, y


    def draw_random_sequences(self, num_return_sequences:int, sentences_length:int, enc:tiktoken.core.Encoding)->torch.Tensor:
        #pick random sentence(s) for starting validation evaluation
        B,T = num_return_sequences, sentences_length
        buf = self.tokens[0:B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        random_indice = torch.randint(low=0, high=B, size=()).item()
        sentence = x[random_indice]
        sentence = [t for t in sentence if 0 <= t < enc.n_vocab]  # filter out out of bound tokens
        decoded_sentence = enc.decode(sentence)
        return decoded_sentence





#upload raw data saved as shards
def upload_shards(filename:str)->numpy.ndarray:
    shard = np.load(filename)
    # npt = npt.astype(np.int32) # added after video
    # ptt = torch.tensor(npt, dtype=torch.long)
    return shard

def do_IO_ops(file_path:str, mode,encoding='utf-8',content=None):
    texts = ''
    if mode == 'a' and content is not None:
        with open(file_path, mode, encoding) as file:
            file.write(content)
            print(f"File written to: {file_path}")
    if mode == 'r':
        with open(file_path, mode, encoding) as f:
            texts = f.read()
            print(f"File read and return from: {file_path}")
    return texts


def clean_dataset(texts:str)->list:
    lst = []
    for phrase in texts:
        lst.append(f'<|startoftext|> {phrase} <|endoftext|>')
    return lst





# hf_repos = {
#     'nietzsche': 'krasaee/nietzsche',
#     'fr_wiki' : 'Kant1/French_Wikipedia_articles',
#     'fr_wikbooks': 'Kant1/French_Wikibooks_articles',
#     'fr_wikiversity' :'Kant1/French_Wikiversity_articles' ,
#     'ft_wiki_fineTuning' : 'Sabrina1763/wikipedia_french',
#     'fr_wiki_trivia_fineTuning': 'AIffl/french_trivia_qa_with_wikicontext'
# }
# train_split, val_split, test_split = 0.80, 0.15, 0.5
# enc = tiktoken.get_encoding("gpt2")
# B, T = 4, 32 # Batch and sequence length


#---------------------------------------------------------------
#get & cache raw text from hggf
# dataset = download_and_cache_dataset(hf_repos['nietzsche'])
# datasets = split_dataset(dataset,  train_split,val_split, test_split)

#tokenize and save datasets
# for key, value in datasets.items():
#     tokenized_dataset = tokenize(enc,value)
#     dataset_path = get_dataset_path(key)
#     save_data_shards(dataset_path,tokenized_dataset)

#load tokenized datasets
# tokenized_datasets = {}
# for key, _ in datasets.items():
#     tokenized_datasets[key] = load_tokens(get_dataset_path(key))
#     #print(f'{key} : {len(tokenized_datasets[key])}')

#get datasets loaders
# dataset_loaders = {}
# dataset_loaders['train_loader'] = DataLoaderLite(B=B, T=T, process_rank=0,num_processes=1,tokenized_dataset=tokenized_datasets['train_set'] )
# dataset_loaders['val_loader'] = DataLoaderLite(B=B, T=T, process_rank=0,num_processes=1,tokenized_dataset=tokenized_datasets['val_set'] )
# dataset_loaders['test_loader'] = DataLoaderLite(B=B, T=T, process_rank=0,num_processes=1,tokenized_dataset=tokenized_datasets['test_set'] )

# print("dataset_loaders", dataset_loaders)



#save and load raw untokenized data
#save_data_shards(data_file_path, dataset, False)           #save raw data into shards
#shards = upload_shards(data_file_path)                    #upload raw shards






#train_loader = DataLoaderLite(B=B, T=T, process_rank=0,num_processes=1,file_path=file_path ) #B=4 T=32



