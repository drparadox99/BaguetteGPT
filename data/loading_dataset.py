import pandas as pd
import torch
import ast
import numpy
import multiprocessing as mp
import os
import numpy as np
import tiktoken
from datasets import load_dataset,Dataset # pip install datasets
from itertools import islice
from typing import Generator, Any
from tqdm import tqdm # pip install tqdm (progress bar)
import tiktoken


"""
krasaee/nietzsche dataset (for srs pretraining)
https://huggingface.co/datasets/krasaee/nietzsche
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python load_dataset.py
Will save shards to the local directory "nietzsche_shards".
"""



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
    np.save(file_path, texts)
    print(f"INFO: {file_path} written ! (size {len(texts)})")

#save tokenized datasets(train,val and testing)
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
    def __init__(self, B:int, T:int, process_rank:int, num_processes:int,remote_data_path:str, split:str,master_process:bool):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # get the shard filenames
        data_root = remote_data_path
        shards = os.listdir(data_root) #list context of folder
        #filter out files not associated with split (val or train)
        shards = [s for s in shards if split in s] 
        shards = sorted(shards) #sort shards
        #get shards full path of associated shards
        shards = [os.path.join(data_root, s) for s in shards] 
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
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
            self.current_shard = (self.current_shard + 1) % len(self.shards) #wraps around to 0 if it exceeds the last index (like circular rotation).
            self.tokens = load_tokens(self.shards[self.current_shard])
            print("Reached the end of the datast, reseting index...")
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
    
    

# #DataLoaderLite for multiple processes GPUs
# class DataLoaderLite:
#     def __init__(self, B:int, T:int, process_rank:int, num_processes:int,tokenized_dataset:torch.Tensor):
#         self.B = B
#         self.T = T
#         self.process_rank = process_rank
#         self.num_processes = num_processes

#         self.tokens = tokenized_dataset #load_tokens(file_path)
#         self.tokenized_dataset = tokenized_dataset
#         print(f"INFO(Dataset split): loaded {len(self.tokens)} tokens")  #2288478 tokens
#         # starting point for each process(GPU)
#         self.current_position = self.B * self.T * self.process_rank #first process starts at pos 0

#     #next batch for each GPU process

#     def reset(self):
#         # state, init at shard zero
#         self.current_shard = 0
#         self.tokens = self.tokenized_dataset #load_tokens(self.shards[self.current_shard])
#         self.current_position = self.B * self.T * self.process_rank

#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position : self.current_position+B*T+1]
#         x = (buf[:-1]).view(B, T) # inputs
#         y = (buf[1:]).view(B, T) # targets
#         # advance the position in the tensor
#         self.current_position += B * T * self.num_processes #each process advances by the entire chunk to get next batch
#         # if loading the next batch would be out of bounds, advance to next shard
#         if self.current_position + (B * T * self.num_processes+ 1) > len(self.tokens):
#             print("Reached the end of the datast, reseting index...")
#             self.current_position = self.B * self.T * self.process_rank
#         return x, y


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

def stream_yield_dataset_in_chunks(dataset_name:str, split:str, chunk_size:int, max_chunks=None)-> Generator[Dataset, Any, None]:
    """
    Streams a dataset from Hugging Face and yields it in chunks.
    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub
        split (str): Which split to stream (default: "train").
        chunk_size (int): How many samples to fetch per chunk.
        max_chunks (int): Optional limit on number of chunks to yield.
    Yields:
        datasets.Dataset: A chunk of data as a Hugging Face Dataset.
    """
    #streamed_data = load_dataset(dataset_name,'wikitext-103-raw-v1', split=split, streaming=True)
    streamed_data = load_dataset(dataset_name, split=split, streaming=True)
    iterator = iter(streamed_data)

    chunk_index = 0
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break  # no more data
        yield Dataset.from_list(chunk)
        chunk_index += 1
        if max_chunks is not None and chunk_index >= max_chunks:
            break

def download_dataset_in_chunks(remote_data_path:str,shard_size:int,chuck_size,enc:tiktoken.core.Encoding, num_shards_allowed=None)->None:
    shard_index = 0
    token_count = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = None

    for i, chunk in enumerate(stream_yield_dataset_in_chunks(os.path.relpath(remote_data_path, "data"), split="train", chunk_size=chuck_size)):

        chunk_tokens = tokenize(enc,chunk['text']) #tokenize texts

        if token_count + len(chunk_tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(chunk_tokens)] = chunk_tokens
            token_count += len(chunk_tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(chunk_tokens))            
        else:
            # write the current shard and start a new one
            # os.makedirs(os.path.dirname(local_path), exist_ok=True)
            #split = "val_set" if shard_index == 0 else "train_set"
            split = "train_set"
            os.makedirs(os.path.dirname(remote_data_path), exist_ok=True)
            # os.makedirs(os.path.join(os.path.dirname(__file__),remote_data_path), exist_ok=True)
            filename = f"{remote_data_path}{split}_shard_{shard_index:06d}.npy"     
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = chunk_tokens[:remainder]
            save_data_shards(filename,all_tokens_np)  
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(chunk_tokens)-remainder] = chunk_tokens[remainder:]
            token_count = len(chunk_tokens)-remainder                       

         # write any remaining tokens as the last shard
    if token_count != 0:
        #split = "val_set" if shard_index == 0 else "train_set"
        split = "val_set" 
        filename = f"{remote_data_path}{split}_shard_{shard_index:06d}.npy"
        save_data_shards(filename,all_tokens_np[:token_count])   



#Downlaod datasets in data directory
if __name__ == "__main__":

    #some huggingface repos
    hf_repos = {
    'nietzsche': 'krasaee/nietzsche/',
    'fr_wiki' : 'Kant1/French_Wikipedia_articles/',
    'fr_wikbooks': 'Kant1/French_Wikibooks_articles/',
    'fr_wikiversity' :'Kant1/French_Wikiversity_articles/' ,
    'ft_wiki_fineTuning' : 'Sabrina1763/wikipedia_french/',
    'fr_wiki_trivia_fineTuning': 'AIffl/french_trivia_qa_with_wikicontext/'
    }

    enc = tiktoken.get_encoding("gpt2")
    shard_size = int(1e6)#int(1e8)
    remote_data_path = 'data/'+hf_repos['fr_wiki']#"data/krasaee/nietzsche/"
    num_shards_allowed = 5
    chunck_size = 100



    #remote_name = "sample-10BT"
    download_dataset_in_chunks(remote_data_path,shard_size,chunck_size,enc)


    #command for lauching file 
    #python3 data/loading_dataset.py

