import os
import time
import torch
import torch.distributed as dist
import argparse
import torch
from torch.multiprocessing import spawn


# ============================================================
# 1. Distributed environment initialization and cleanup
# ============================================================

def init_distributed_environment(
    master_addr: str,
    master_port: int,
    global_rank: int,
    world_size: int,
    backend: str
):
    """
    Initialize the PyTorch distributed communication environment (process group).

    Args:
    - master_addr : IP address of the node where rank 0 runs
    - master_port : Port used for inter-process communication
    - global_rank : Unique ID of the current process among all processes
    - world_size  : Total number of processes
    - backend     : Communication backend (gloo / nccl / mpi)
    """

    # --------------------------------------------------------
    # 1.1 Set rendezvous address (where all processes meet)
    # --------------------------------------------------------
    # All processes must establish the initial connection via MASTER_ADDR:MASTER_PORT
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # --------------------------------------------------------
    # 1.2 Initialize process group (the "master switch" of distributed training)
    # --------------------------------------------------------
    # This is a prerequisite for using torch.distributed
    dist.init_process_group(
        backend=backend,
        rank=global_rank,
        world_size=world_size
    )


def destroy_distributed_environment():
    """
    Clean up the distributed environment and release resources
    """

    # Destroy the process group to prevent resource leaks or deadlocks
    dist.destroy_process_group()

    # On GPU, clear PyTorch's CUDA cache (optional but recommended)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 2. All-Reduce communication performance benchmark
# ============================================================

def run_all_reduce_benchmark(
    global_rank: int,
    world_size: int,
    tensor_size_mb: int,
    backend: str,
    device: str,
    master_addr: str,
    master_port: int
):
    """
    Benchmark dist.all_reduce performance.

    Metrics:
    - Average latency per all-reduce
    - Theoretical communication bandwidth (GB/s)
    """

    # --------------------------------------------------------
    # 2.1 Initialize distributed environment
    # --------------------------------------------------------
    init_distributed_environment(
        master_addr=master_addr,
        master_port=master_port,
        global_rank=global_rank,
        world_size=world_size,
        backend=backend
    )

    # --------------------------------------------------------
    # 2.2 Bind to GPU (only in CUDA scenarios)
    # --------------------------------------------------------
    if device == "cuda":
        # Convention: rank i -> GPU i
        torch.cuda.set_device(global_rank)
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 2.3 Construct the tensor used for communication testing
    # --------------------------------------------------------
    # Convert MB to bytes
    tensor_num_bytes = tensor_size_mb * 1024 * 1024

    # float32 takes 4 bytes
    bytes_per_element = 4
    num_elements = tensor_num_bytes // bytes_per_element

    # Random test data (the actual values do not matter)
    communication_tensor = torch.randn(
        num_elements,
        device=device,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # 2.4 Warm-up (very important)
    # --------------------------------------------------------
    # Reasons:
    # - NCCL communicator initialization
    # - CUDA kernel lazy initialization
    # - GPU frequency scaling
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        dist.all_reduce(
            communication_tensor,
            op=dist.ReduceOp.SUM
        )

        # CUDA executes asynchronously, so explicit synchronization is required
        if device == "cuda":
            torch.cuda.synchronize()

    # Ensure all processes start the formal test at the same moment
    dist.barrier()

    # --------------------------------------------------------
    # 2.5 Formal benchmark (timing)
    # --------------------------------------------------------
    benchmark_iterations = 20

    start_time = time.time()

    for _ in range(benchmark_iterations):
        dist.all_reduce(
            communication_tensor,
            op=dist.ReduceOp.SUM
        )

    # Wait for all kernels to finish on GPU
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # --------------------------------------------------------
    # 2.6 Performance metric calculation
    # --------------------------------------------------------
    total_elapsed_time = end_time - start_time
    avg_latency_seconds = total_elapsed_time / benchmark_iterations

    # Bandwidth = data size / time (in GB/s)
    bandwidth_gbps = (tensor_num_bytes / avg_latency_seconds) / 1e9

    # --------------------------------------------------------
    # 2.7 Print results (rank 0 only)
    # --------------------------------------------------------
    if global_rank == 0:
        print(
            f"[All-Reduce Benchmark]\n"
            f"  Backend        : {backend}\n"
            f"  Device         : {device}\n"
            f"  World Size     : {world_size}\n"
            f"  Tensor Size    : {tensor_size_mb} MB\n"
            f"  Avg Latency    : {avg_latency_seconds * 1000:.4f} ms\n"
            f"  Bandwidth      : {bandwidth_gbps:.4f} GB/s\n"
        )

    # --------------------------------------------------------
    # 2.8 Gather results from all ranks (for teaching)
    # --------------------------------------------------------
    local_result = {
        "rank": global_rank,
        "world_size": world_size,
        "backend": backend,
        "device": device,
        "tensor_size_mb": tensor_size_mb,
        "avg_latency_ms": avg_latency_seconds * 1000,
        "bandwidth_gbps": bandwidth_gbps,
    }

    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)

    # --------------------------------------------------------
    # 2.9 Clean up environment
    # --------------------------------------------------------
    destroy_distributed_environment()

    # Only the main process returns results
    if global_rank == 0:
        return gathered_results


def main():
    # Fixed run-level settings, parsed once (not re-parsed per sweep iteration -
    # re-parsing sys.argv inside the loop would let one CLI invocation silently
    # override every point of the sweep below).
    parser = argparse.ArgumentParser("All-Reduce Benchmark")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    base_args = parser.parse_args()

    for size in [1,10,100,1000]:
        for Backend in ["gloo", "nccl"]:
            for world_size in [1]:
                args = argparse.Namespace(
                    world_size=world_size,
                    tensor_size_mb=size,
                    backend=Backend,
                    device=base_args.device,
                    master_addr=base_args.master_addr,
                    master_port=base_args.master_port,
                )

                # GPU count check
                if args.device == "cuda":
                    assert torch.cuda.is_available(), "CUDA not available"
                    assert args.world_size <= torch.cuda.device_count(), (
                        "world_size cannot exceed the number of GPUs"
                    )

                # Launch multi-process using spawn
                spawn(
                    fn=run_all_reduce_benchmark,
                    args=(
                        args.world_size,
                        args.tensor_size_mb,
                        args.backend,
                        args.device,
                        args.master_addr,
                        args.master_port
                    ),
                    nprocs=args.world_size,
                    join=True
                )


if __name__ == "__main__":
    main()

