import os
import torch

local_rank = int(os.environ['SLURM_PROCID']) % int(os.environ['SLURM_JOB_NUM_NODES'])
world_rank = int(os.environ['SLURM_PROCID'])

torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=world_rank)
print(torch.distributed.get_rank())

test = torch.tensor(torch.distributed.get_rank()).cuda(local_rank)
torch.distributed.all_reduce(test)

print(torch.distributed.get_rank(), test)

