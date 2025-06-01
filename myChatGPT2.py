
from data.loading_dataset import DataLoaderLite,download_and_cache_dataset,split_dataset,save_tokenized_datasets,load_tokenized_datatsets
from models.model import get_model
from utils.utils import  enable_ddp, get_device,get_lr,destroy_ddp,evaluate_running_time
from utils.EarlyStopping import  EarlyStopping
import torch
from torch.nn import functional as F
import time
import torch.distributed as dist
import os
import tiktoken

#useful variables
# file_path = "data/nietzsche_shards/nietzsche_train_000000.npy"  # data path
file_path = "data/nietzsche_shards/nietzsche"
device = get_device()  #get device
compiled_model = False
sample_file_="logs/sample_log.txt"
enc = tiktoken.get_encoding("gpt2")
val_log_file = "logs/val_log.txt"
checkpoint_path = "checkpoints/checkpoint.pt" #f"model_{step:05d}.pt"
load_checkpoint = False  #upload checkpoint model
save_checkpoint = False

#training hyperparameters
B, T = 4, 32 # Batch and sequence length
num_return_sequences, sentences_length = 5, 7
epochs = 200
max_length = 50
weight_decay=0.1
learning_rate=6e-4
total_batch_size = 16384 #(2**11) # 2**19=524288, ~0.5M, in number of tokens in a single batch
val_occurrence = 2 #10
sampling_occurrence = 10
checkpoint_occurrence = 2 #30
chatGPT_name = "PhiloGPT"

#early stopping
patience = 10
delta = 0.0
verbose = True
early_stopping = EarlyStopping(patience=patience, delta=delta, device=device, verbose=verbose)

#datasets split
#train_split, val_split, test_split = 0.80, 0.15, 0.5
datasets_splits = {
    "train_set": 0.80,
    "val_set": 0.15,
    "test_set": 0.5,
}


#datasets repos
hf_repos = {
    'nietzsche': 'krasaee/nietzsche',
    'fr_wiki' : 'Kant1/French_Wikipedia_articles',
    'fr_wikbooks': 'Kant1/French_Wikibooks_articles',
    'fr_wikiversity' :'Kant1/French_Wikiversity_articles' ,
    'ft_wiki_fineTuning' : 'Sabrina1763/wikipedia_french',
    'fr_wiki_trivia_fineTuning': 'AIffl/french_trivia_qa_with_wikicontext'
}

##Prepare datasets (training, validation and testing)

#get & cache raw text from hggf
# dataset = download_and_cache_dataset(hf_repos['nietzsche'])
# datasets = split_dataset(dataset,  datasets_splits['train_set'],datasets_splits['val_set'], datasets_splits['test_set'])

#save datasets
# save_tokenized_datasets(datasets, enc,file_path)

#load datasets
tokenized_datasets = load_tokenized_datatsets(file_path, datasets_splits)

#prepare model
model = get_model(pretrained=False)
model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, device_type=device) #warmed up lr

#upload checkpoint
model = early_stopping.loadModel(model, checkpoint_path, optimizer, 'train')  if load_checkpoint else model

#compile model
model = torch.compile(model) if compiled_model else model

#enable parallel data distributed if available # use torchrun --standalone --nproc_per_node=6 myChatGPT2.py
model, ddp, ddp_rank, ddp_local_rank,ddp_world_size,master_process   = enable_ddp(model,device)
chatGPT_model = model.module if ddp else model #get model if ddp enabled

#prepare dataloaders
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size,tokenized_dataset=tokenized_datasets['train_set'])
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size,tokenized_dataset=tokenized_datasets['val_set'])
test_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,num_processes=ddp_world_size,tokenized_dataset=tokenized_datasets['test_set'])

#pick random sentene in testing data
random_sentence = test_loader.draw_random_sequences(num_return_sequences,sentences_length, enc)

#set (lower) precision to tensorfloat32('high')
torch.set_float32_matmul_precision('high')

#grad accumulation parameters
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) #number of single-backward operations before a single update
if master_process:
    print(f"INFO: total desired batch size: {total_batch_size}")
    print(f"INFO: calculated gradient accumulation steps:  {grad_accum_steps}")




 # ----------------------------------------Validation-Samping-Training loop----------------------------------------------------


ftime_s = time.time()
for step in range(epochs):
    t0 = time.time() #in seconds
    last_epoch = (step == epochs - 1)

    # ----------------------------------------validation loop----------------------------------------------------
    if step % val_occurrence == 0 and step != 0 or last_epoch:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"INFO(Validation loss): validation loss: {val_loss_accum.item():.4f}")
            with open(val_log_file, "a") as f:
                f.write(f"step: {step}, val_loss: {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % checkpoint_occurrence == 0 or last_epoch): #to remove
                early_stopping(chatGPT_model,checkpoint_path,optimizer, epochs, step,val_loss_accum)
        #early stopping
            if save_checkpoint:
                early_stopping(chatGPT_model,checkpoint_path,optimizer, epochs, step,val_loss_accum)
                if early_stopping.early_stop:
                    break


    # ----------------------------------------sampling loop----------------------------------------------------
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % sampling_occurrence == 0) or last_epoch) and (not compiled_model):
        model.eval()
        max_length = 32
        tokens = enc.encode_ordinary(random_sentence)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #repeat to across time dim (num_return_sequences times)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                sampled_token_ids = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                #Sanitize token IDs: clamp to valid range (0 to vocab_size-1)
                sampled_token_ids = torch.clamp(sampled_token_ids, min=0, max=enc.n_vocab - 1) #
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, sampled_token_ids) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        sampled_text = [f"\n*****************Sampling from the model (step:{step})*****************", "\n"]
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            tokens = [t for t in tokens if 0 <= t < enc.n_vocab] #filter out out of bound tokens
            decoded = enc.decode(tokens)
            decoded = f'({chatGPT_name}):{decoded} '
            sampled_text.append(decoded + "\n")
            if master_process and i == 0:
                print(sampled_text[0]+'\n') #sampling announcement
            print(f"rank {ddp_rank} sample {i}: {decoded}")
        with open(sample_file_, "a",encoding='utf-8') as file:
            file.write('\n'.join(sampled_text)+ '\n')

    #----------------------------------------training loop----------------------------------------------------
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps): #larger batch sizes can stabilize training or improve convergence.
        x,y  = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        if ddp:  # average all gradients across processes only at the last step of grad_accum_steps
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) #if DDP False skips gradient synchronization during backward().
        if device == 'cpu':
            logits, loss = model(x, y)
        else:
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #drop to float16
                logits, loss = model(x,y)
        loss = loss / grad_accum_steps #get mean of gradients, due to gradient accumulation implementation
        loss_accum += loss.detach() #get overall loss across the loss_accum
        loss.backward()
    if ddp: #loss_accum tensor exists on all the ranks, when all_reduce is called, it averages all tensors across all processes and deposits the aveage on the processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    #gradient clipping (calculate the global norm of the parameters)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    #determine and set the learning rate for this itaration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()  #update weights only after averaging loss acrrosss all processes and after grand_accum_steps
    #torch.cuda.synchronize() #wait for the GPU to finish before executing the following instruction
    t1 = time.time()
    dt = (t1 - t0)* 1000 #time difference in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size #num of tokens processed across all processes
    print("INFO(training): Tokens_processed:", tokens_processed)
    tokens_per_sec = tokens_processed / (t1 - t0)
    if master_process:
        print(f"\nINFO(master process): | step: {step} | loss: {loss_accum.item()} | norm: {norm:.4f} | lr {lr:.4e} | dt: {dt:.2f}ms | tok/sec: {tokens_processed:.2f}")
f_time_e = time.time()
print(f"final dt: {(f_time_e-ftime_s):.2f}s")

destroy_ddp(ddp)

import sys;sys.exit(0)




#Initial chatGPT configuration :
    # Specific way of initializing the modules and sub-modules of the model -> init_weights()
    # Control the growth of activations inside the residual stream in a forward pass -> NANOGPT_SCALE_INIT
    #For GPUs
    # Reduce precision to tensorfloat32 (crops up precision/mantissag to m10) instead of float32 (exponent(8) + mantissa(23) + sign (1) = 32), but keeps range/exponent intact
    # Reduce precision to BFloat16 (crops up precision/mantissag to m7), float32 (exponent(8) + mantissa(23) + sign (1) = 32), but keeps range/exponent intact (e8)
    # Complile code for exucution by using torch.compile
    # Flash attention
    # Change numbers in code to powers of two
    #use specific chatGPT hyperparameters values (decayable lr, betas, gradient clipping, lr scheduler, weight decay ...)
    #During training data are sampled without replacement (already implemented)
    #gradient accumulation (simulate arbitrary batch size)
    #Parallel computation (distributed data parallel (DDP))
    #using fused



    #commands used
        # nvidia-smi (dispaly available gpu)
        # import code;code.interact(local=locals())(for debugging via terminal)



#des choses à ajouter
#add early stopping (monitor loss and valid loss) save ending training (fait)
#mixture of experts
#At every epoch shuffle examples in training dataloader to introduce randomness

#Des choses à revoir
#llama 4
#position embedding :
#token embedding:
#.attn.masked_bias: #buffers (none trainable parameters)
#.attn.bias (fait): #buffers (none trainable parameters)
#KV cache VS Multi-Head Latent Attention (deepseek)
#Multi-headed vs Grouped-Query attention vs MultiQuery attention vscale
#Quantization
#LoRA
#OpenRouter
#Mixture of experts
#gradient clipping
