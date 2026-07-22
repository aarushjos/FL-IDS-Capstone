from src.components.data.data_partitioner import (
    load_partition_dataloaders,
)

import torch


train_loader,_=load_partition_dataloaders(
    client_id=0,
    batch_size=256
)

x,y=next(iter(train_loader))

print("X shape =",x.shape)
print("Unique labels =",torch.unique(y))
print("Max label =",torch.max(y))
print("Min label =",torch.min(y))