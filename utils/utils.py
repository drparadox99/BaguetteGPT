
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP, DistributedDataParallel
import math
import os
import torch.distributed as dist


# def enable_ddp(model: any, device: str) -> tuple[DDP, bool, int, int, int, bool]:
#     print(f"[Rank {os.environ.get('RANK', '?')}] LOCAL_RANK={os.environ.get('LOCAL_RANK', '?')} - Available devices: {torch.cuda.device_count()}")

#     ddp = int(os.environ.get('RANK', -1)) != -1
#     if ddp:
#         assert torch.cuda.is_available(), "CUDA required for DDP"
#         init_process_group(backend='nccl')

#         ddp_rank = int(os.environ['RANK'])
#         ddp_local_rank = int(os.environ['LOCAL_RANK'])
#         ddp_world_size = int(os.environ['WORLD_SIZE'])

#         device = f'cuda:{ddp_local_rank}'
#         torch.cuda.set_device(ddp_local_rank)
#         model = model.to(device)

#         master_process = ddp_rank == 0
#         model = DDP(model, device_ids=[ddp_local_rank])  # ✅ FIXED

#         if master_process:
#             print("Master process : DDP enabled")
#     else:
#         ddp_rank = 0
#         ddp_local_rank = 0
#         ddp_world_size = 1
#         master_process = True
#         print("INFO: DDP is not enabled")
#         model = model.to(device)

#     return model, ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process

#enables parallel processing if available
def enable_ddp(model:any,device:str)-> tuple[DistributedDataParallel, bool, int, int, int, bool]:
	#set up DDP (distributed data parallel).
	# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
	print(f"[Rank {os.environ.get('RANK', '?')}] LOCAL_RANK={os.environ.get('LOCAL_RANK', '?')} - Available devices: {torch.cuda.device_count()} \n")

	ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
	if ddp:
		# use of DDP atm demands CUDA, we set the device appropriately according to rank
		assert torch.cuda.is_available(), "CUDA required for DDP"
		init_process_group(backend='nccl') #enable GPU-to-GPU communication
		
		ddp_rank = int(os.environ['RANK']) #process rank/identifier (rank/identifier of a GPU)
		ddp_local_rank = int(os.environ['LOCAL_RANK']) #rank of the GPU in a single node 
		ddp_world_size = int(os.environ['WORLD_SIZE']) #number of processes to run in parallel (8 GPUs)

		device_ = f'cuda:{ddp_local_rank}'
		torch.cuda.set_device(ddp_local_rank)
		model = model.to(device_) #multiple GPUs: device= "cuda:0", "cuda:1" but device variable remains device="cuda"
		master_process = ddp_rank == 0 # set first process(rank0) to masterprocess for logging, checkpointing etc.
		model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])]) #DDP wrapper
		if master_process:
				print("Master process : DDP enabled")
	else: #single GPU training
		# vanilla, non-DDP run
		ddp_rank = 0
		ddp_local_rank = 0
		ddp_world_size = 1
		master_process = True
		print("INFO: DDP is not enabled")
		model = model.to(device)

	return model, ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process

def destroy_ddp(ddp)->None:
	if ddp:
		destroy_process_group()


def get_device(device_selected=None)->str:
	device_ = "cpu"
	if torch.cuda.is_available():
		device_ = "cuda"
	elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		device_ = "mps"
	# device = "cpu"
	device = device_ if device_selected is None else device_selected
	print(f"using device: {device}")
	return device


def get_lr(it:int)->float:
	max_lr = 6e-4
	min_lr = max_lr * 0.1
	warmup_steps = 715
	max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
	# 1) linear warmup for warmup_iters steps
	if it < warmup_steps:
		return max_lr * (it + 1) / warmup_steps
	# 2) if it > lr_decay_iters, return min learning rate
	if it > max_steps:
		return min_lr
	# 3) in between, use cosine decay down to min learning rate
	decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
	return min_lr + coeff * (max_lr - min_lr)


def evaluate_running_time(epochs:int,time_per_epoch:float)->None:
    milliSec_to_sec_per_epoch = time_per_epoch / 1000
    time_in_sec = epochs * milliSec_to_sec_per_epoch
    time_in_min = time_in_sec / 60
    time_in_hr = time_in_min / 60
    print(f"INFO: exec time = min: {time_in_min} - hr: {time_in_hr}")

# def get_checkpoint_path(step:str)->str:
# 	return f'checkpoints/checkpoint_{step:05d}.pt'

# Check if bfloat16 is supported on current device
def is_bfloat16_supported(device_type):
    if device_type == 'cuda':
        # bfloat16 supported only on Ampere GPUs (compute capability >= 8.0)
        # You can infer it using torch.cuda.get_device_capability
        major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        return (major >= 8)


