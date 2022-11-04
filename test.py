import models
import torch 
import torch.distributed as dist
import os 
import torch.nn.functional as F
import time

torch.backends.cudnn.benchmark = True
use_pytorch_ddp = 'LOCAL_RANK' in os.environ
rank = int(os.environ['LOCAL_RANK']) if use_pytorch_ddp else 0
device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
torch.cuda.set_device(rank)
if use_pytorch_ddp:
    dist.init_process_group('nccl')
EX_PER_DEVICE = 1024//8

# Initialize
m = models.resnet50()

# Move to cuda 
if use_pytorch_ddp:
    m = torch.nn.parallel.DistributedDataParallel(m.cuda(rank),device_ids=[rank], output_device=rank).train()
    batch_size = EX_PER_DEVICE
else:
    m = torch.nn.DataParallel(m.cuda()).train()
    batch_size = EX_PER_DEVICE * n_gpus
opt = torch.optim.Adam(m.parameters())

# Create Dataset
USE_PYTORCH_DDP = use_pytorch_ddp
train = True
N_GPUS = n_gpus 
RANK = rank 

def train_step():
    opt.zero_grad(set_to_none=True)
    inp = torch.randn(size=(batch_size, 3, 224, 224))
    logits = m(inp.to(device))
    if rank==0:
        print(f"Memory allocated after fwd: ",torch.cuda.memory_allocated()/1024**3,"GB")

    loss = F.cross_entropy(logits, torch.randint(low=0, high=1000, size=(logits.shape[0],)).cuda(rank)).mean()
    loss.backward()
    opt.step()
    return loss.sum().item()


# 10 step of training
for i, batch in enumerate(range(10)):
    if rank==0:
        print(f"Memory allocated: ",torch.cuda.memory_allocated()/1024**3,"GB")
    t = time.time()
    loss = train_step()
    if rank==0:
        print(f"{loss}")
        print(time.time()-t)
