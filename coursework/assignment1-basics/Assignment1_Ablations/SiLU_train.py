import argparse
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import psutil  # requires: pip install psutil
import torch
from SiLU_model import *
# Training environment initialization (prevents OMP conflicts)
import os
from transformers import PreTrainedTokenizerFast

# Allow duplicate OpenMP runtime (prevents libiomp5md.dll conflicts)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Limit OpenMP threads to prevent multi-threading conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Optional: display current thread settings for debugging
#print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
#print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")

class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None: continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Correct bias correction logic
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                denom = exp_avg_sq.sqrt().add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)
                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)
        return loss


def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine_out = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_out * (alpha_max - alpha_min)
    else:
        return alpha_min


def run_gradient_clipping(params, max_norm, eps=1e-6):
    params_with_grad = [p for p in params if p.grad is not None]
    if len(params_with_grad) == 0: return
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2)
    clip_coeff = max_norm / (total_norm + eps)
    if clip_coeff < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)


# Part 4: Data loading
class CausalMemmapDataset(Dataset):
    def __init__(self, data_path, context_length, start_block=0, end_block=None):

        # Ensure dtype consistency; int32 is usually enough for corpus indices
        self.data = np.memmap(data_path, mode='r', dtype=np.int32)
        self.context_length = context_length

        total_blocks = (len(self.data) - context_length - 1) // context_length

        if end_block is None:
            end_block = total_blocks

        self.start_block = start_block
        self.end_block = end_block
        self.num_blocks = end_block - start_block

        # Simple boundary check
        if self.num_blocks <= 0:
            print(f"Warning: Dataset has 0 blocks. (Start: {start_block}, End: {end_block})")

    def __len__(self):
        return max(0, self.num_blocks)

    def __getitem__(self, idx):
        block_idx = self.start_block + idx
        start_idx = block_idx * self.context_length

        # Convert to int64 for PyTorch
        x = torch.from_numpy(
            self.data[start_idx: start_idx + self.context_length].astype(np.int64)
        )

        y = torch.from_numpy(
            self.data[start_idx + 1: start_idx + self.context_length + 1].astype(np.int64)
        )

        return x, y


def save_ppl_curve(train_ppls, val_ppls, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Step-level PPL
    plt.figure()
    plt.plot(train_ppls)
    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexity (per step)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_ppl3.png"))
    plt.close()

    # Validation PPL
    plt.figure()
    plt.plot(val_ppls)
    plt.yscale("log")
    plt.xlabel("Val Step")
    plt.ylabel("Perplexity")
    plt.title("Val Perplexity (per step)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_ppl3.png"))
    plt.close()


# Part 5: Training loop
def save_checkpoint(
    path,
    model,
    optimizer,
    iteration,
    epoch,
    config: dict
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
        "config": config,
    }

    torch.save(ckpt, path)

def get_memory_usage(device):
    """Get current memory / VRAM usage"""
    if device == "cuda":
        # Get allocated VRAM on current device (MB)
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        return f"{mem:.2f} MB (GPU)"
    else:
        # Get system memory used by the current process (MB)
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2
        return f"{mem:.2f} MB (CPU)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="./Assignment1_Ablations/SiLU")
    parser.add_argument("--data_path", type=str, default="data.in")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Generate dummy data (dtype must be int32 to match Dataset reading)
    if not os.path.exists(args.data_path):
        print(f"Creating dummy data {args.data_path}...")
        # Ensure enough data to generate several batches
        dummy_len = args.context_length * args.batch_size * 20
        dummy_data = np.random.randint(0, args.vocab_size, (dummy_len,), dtype=np.int32)
        dummy_data.tofile(args.data_path)

    total_ds = CausalMemmapDataset(args.data_path, args.context_length)
    total_blocks = len(total_ds)
    split = 0.8
    split_block = int(total_blocks * split)

    train_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=0,
        end_block=split_block
    )

    val_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=split_block,
        end_block=total_blocks
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json"
    )

    vocab_size = tokenizer.vocab_size

    # Ensure dataset is not empty
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Increase the data size or reduce context_length.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # Statistics containers
    train_ppls = []
    val_ppls = []

    # Epoch-level statistics
    epoch_avg_tlosses = []
    epoch_avg_tppls = []
    epoch_avg_vlosses = []
    epoch_avg_vppls = []

    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = CustomAdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    step_t = 0
    step_v = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()  # Record start time
        model.train()  # Enable training mode

        current_epoch_losses = []  # Used to compute the average loss of the current epoch
        print(f"\n--- Epoch {epoch} ---")
        print(f"Initial memory usage: {get_memory_usage(device)}")

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr_cosine_schedule(step_t, args.lr, args.min_lr, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            logits = model(x)  # (B, T, Vocab)

            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # Variable definition
            loss_val = loss.item()
            ppl_val = math.exp(loss_val)

            loss.backward()
            run_gradient_clipping(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record step-level metrics
            current_epoch_losses.append(loss_val)
            train_ppls.append(ppl_val)
            step_t += 1

            if batch_idx % 100 == 0:  # Reduce print frequency
                mem_status = get_memory_usage(device)
                print(f"Step {batch_idx} | Real-time memory: {mem_status}")
                print(f"Epoch {epoch} | Step {step_t}/{total_steps} | "
                      f"LR: {lr:.6f} | Train Loss: {loss_val:.4f} | Train PPL: {ppl_val:.4f}")

        # Epoch-end statistics
        if len(current_epoch_losses) > 0:
            epoch_tloss = sum(current_epoch_losses) / len(current_epoch_losses)
            epoch_tppl = math.exp(epoch_tloss)
        else:
            epoch_tloss, epoch_tppl = 0.0, 0.0
        epoch_train_end_time = time.time()
        train_duration = epoch_train_end_time - epoch_start_time

        # Compute and print training statistics
        epoch_tloss = sum(current_epoch_losses) / len(current_epoch_losses) if current_epoch_losses else 0
        print(f"[Epoch {epoch} training finished] Time: {train_duration:.2f}s | "
              f"Avg Loss: {epoch_tloss:.4f} | Memory usage: {get_memory_usage(device)}")
        epoch_avg_tlosses.append(epoch_tloss)
        epoch_avg_tppls.append(epoch_tppl)
        print(f"[Epoch {epoch} END] Train Avg Loss: {epoch_tloss:.4f} | Train PPL: {epoch_tppl:.2f}\n")

        # Validation
        val_start_time = time.time()
        model.eval()
        val_losses = []  # Initialize list

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_losses.append(loss.item())
                step_v += 1
                val_ppl = math.exp(loss.item())

                val_ppls.append(val_ppl)

            if len(val_losses) > 0:
                epoch_vloss = sum(val_losses) / len(val_losses)
                epoch_vppl = math.exp(epoch_vloss)
            else:
                epoch_vloss, epoch_vppl = 0.0, 0.0

            epoch_avg_vlosses.append(epoch_vloss)
            epoch_avg_vppls.append(epoch_vppl)

            val_duration = time.time() - val_start_time
            epoch_vloss = sum(val_losses) / len(val_losses) if val_losses else 0
            print(f"[Epoch {epoch} validation finished] Time: {val_duration:.2f}s | "
                  f"Val Loss: {epoch_vloss:.4f} | Memory usage: {get_memory_usage(device)}")

            print(f"[Epoch {epoch} END] Val Avg Loss: {epoch_vloss:.4f} | Val PPL: {epoch_vppl:.2f}")

        save_ppl_curve(
            train_ppls,
            val_ppls,
            args.checkpoint_dir
        )

        if epoch % 10 == 0:

            save_checkpoint(
                path=os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"),
                model=model,
                optimizer=optimizer,
                iteration=step_t,
                epoch=epoch,
                config={
                    "vocab_size": vocab_size,
                    "context_length": args.context_length,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "d_model": args.d_model,
                }
            )

if __name__ == "__main__":
    main()