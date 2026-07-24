import os
import torch
import torch.nn as nn
import torch.distributed as dist

from sharded_optimizer import ShardedOptimizer   # assumes you saved the previous code in this file


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    setup_distributed()

    rank = dist.get_rank()
    device = torch.device("cuda")

    # A simple model
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).to(device)

    # Ensure consistent model initialization
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optimizer = ShardedOptimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    for step in range(5):
        x = torch.randn(8, 1024, device=device)
        y = torch.randint(0, 10, (8,), device=device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()

        # ShardedOptimizer only shards optimizer *state*, not gradient
        # computation - each rank still needs the averaged gradient (as in
        # plain DDP) before step(), otherwise each rank's owned parameters
        # get updated from that rank's own unsynced local gradient only.
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(dist.get_world_size())

        optimizer.step()

        if rank == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
