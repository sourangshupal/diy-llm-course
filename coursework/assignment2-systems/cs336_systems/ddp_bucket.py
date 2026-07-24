import torch
import torch.distributed as dist


# ============================================================
# Part 1. Per-parameter overlap version (the most intuitive DDP teaching implementation)
# ============================================================

class TeachingDDPOverlapPerParam(torch.nn.Module):
    """
    Teaching version of DDP (per-parameter communication):

    - Each parameter triggers an async all-reduce immediately after its backward completes
    - Overlaps computation (backward) with communication
    - Intuitive logic, but the communication granularity is too fine for industry use
    """

    def __init__(self, model: torch.nn.Module):
        # ⚠️ Must initialize the parent class first
        super().__init__()

        self.model = model
        self.world_size = dist.get_world_size()

        # Store handles for all async all-reduces
        self.async_handles = []

        # ----------------------------------------------------
        # Step 1. Initialization: synchronize model parameters & buffers
        # ----------------------------------------------------
        # Equivalent to PyTorch DDP's initial broadcast
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)

            for buffer in self.model.buffers():
                dist.broadcast(buffer.data, src=0)

        # ----------------------------------------------------
        # Step 2. Register backward hooks (core)
        # ----------------------------------------------------
        # Communicate as soon as a parameter's gradient is computed
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._create_grad_hook()
                )

    def _create_grad_hook(self):
        """
        Create a backward hook (closure).

        Hook timing:
        - The current parameter's gradient has been computed
        - But the entire backward pass has not finished yet
        """

        def hook_fn(param: torch.nn.Parameter):
            # Some parameters may have no gradient (e.g., frozen)
            if param.grad is None:
                return

            # ------------------------------------------------
            # Perform async all-reduce on this parameter's gradient
            # ------------------------------------------------
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )

            # Save the handle; wait before the step
            self.async_handles.append(handle)

            # ⚠️ register_post_accumulate_grad_hook must not return any value

        return hook_fn

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def finalize_gradients(self):
        """
        Call before optimizer.step():

        - Wait for all async all-reduces to complete
        - Average the gradients
        """

        # 1. Wait for all communication to finish
        for handle in self.async_handles:
            handle.wait()
        self.async_handles.clear()

        # 2. Average gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(self.world_size)


# ============================================================
# Part 2. Bucket class: the core data structure of DDP
# ============================================================

class GradientBucket:
    """
    A bucket manages gradient communication for a group of parameters:

    - Gradients of multiple parameters are flattened into one contiguous buffer
    - When all parameter gradients in the bucket are ready, trigger one all-reduce
    """

    def __init__(self, params: list[torch.nn.Parameter], bucket_id: int):
        self.bucket_id = bucket_id
        self.params = params

        # Number of parameters already ready
        self.ready_count = 0

        # Communication handle for async all-reduce
        self.comm_handle = None

        # ----------------------------------------------------
        # Step 1. Create a contiguous gradient buffer
        # ----------------------------------------------------
        total_numel = sum(p.numel() for p in params)
        dtype = params[0].dtype
        device = params[0].device

        self.flat_buffer = torch.zeros(
            total_numel, dtype=dtype, device=device
        )

        # ----------------------------------------------------
        # Step 2. Create a buffer view for each parameter
        # ----------------------------------------------------
        self.param_to_view = {}
        offset = 0
        for p in params:
            numel = p.numel()
            self.param_to_view[p] = (
                self.flat_buffer[offset: offset + numel]
                .view(p.shape)
            )
            offset += numel

    def reset(self):
        """
        Call before each forward:
        - Reset the counter
        - Clear the communication handle
        """
        self.ready_count = 0
        self.comm_handle = None


# ============================================================
# Part 3. Bucketized DDP (close to real PyTorch DDP)
# ============================================================

class TeachingDDPBucketed(torch.nn.Module):
    """
    Teaching version of bucketized DDP (industry-grade idea):

    - Parameters are grouped into buckets
    - Each bucket performs one all-reduce
    - Overlap computation and communication during backward
    """

    def __init__(self, model: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        self.model = model
        self.world_size = dist.get_world_size()

        # Bucket size (MB -> Bytes)
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024

        # ----------------------------------------------------
        # Step 1. Initialize parameter synchronization
        # ----------------------------------------------------
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.model.buffers():
                dist.broadcast(buffer.data, src=0)

        # ----------------------------------------------------
        # Step 2. Split parameters into buckets (very important)
        # ----------------------------------------------------
        self.buckets: list[GradientBucket] = []
        self.param_to_bucket: dict[torch.nn.Parameter, GradientBucket] = {}

        # Only consider parameters that require gradients
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        # ⚠️ Backprop order: from output layer to input layer
        # So buckets should be built in "reverse order"
        reversed_params = list(reversed(trainable_params))

        current_bucket_params = []
        current_bucket_bytes = 0

        for param in reversed_params:
            param_bytes = param.numel() * param.element_size()

            # If the current bucket is full, create it first
            if (
                current_bucket_params
                and current_bucket_bytes + param_bytes > self.bucket_size_bytes
            ):
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_bytes = 0

            current_bucket_params.append(param)
            current_bucket_bytes += param_bytes

        # Last bucket
        if current_bucket_params:
            self._create_bucket(current_bucket_params)

        # ----------------------------------------------------
        # Step 3. Register backward hooks
        # ----------------------------------------------------
        for param in trainable_params:
            param.register_post_accumulate_grad_hook(
                self._create_bucket_hook(param)
            )

    def _create_bucket(self, params: list[torch.nn.Parameter]):
        bucket = GradientBucket(params, bucket_id=len(self.buckets))
        self.buckets.append(bucket)

        for p in params:
            self.param_to_bucket[p] = bucket

    def _create_bucket_hook(self, param: torch.nn.Parameter):
        """
        Per-parameter hook:
        - Copy the gradient into the bucket buffer
        - If the bucket is full, launch an async all-reduce immediately
        """

        def hook_fn(p: torch.nn.Parameter):
            if p.grad is None:
                return

            bucket = self.param_to_bucket[p]

            # A. Copy gradient into the bucket buffer
            bucket.param_to_view[p].copy_(p.grad)

            # B. Mark one parameter as ready
            bucket.ready_count += 1

            # C. All parameters in the bucket are ready -> start communication
            if bucket.ready_count == len(bucket.params):
                bucket.comm_handle = dist.all_reduce(
                    bucket.flat_buffer,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )

        return hook_fn

    def forward(self, *args, **kwargs):
        # Reset bucket state before each forward
        for bucket in self.buckets:
            bucket.reset()
        return self.model(*args, **kwargs)

    def finalize_gradients(self):
        """
        Call before optimizer.step():

        - Wait for all bucket communication to complete
        - Average the bucket buffer
        - Copy back to each parameter's p.grad
        """

        for bucket in self.buckets:
            # A bucket whose comm_handle is still None never received all-reduced
            # gradients this step (not all its params got grad) - skip it so we
            # don't divide/copy a stale or un-synced flat_buffer into p.grad.
            if bucket.comm_handle is None:
                continue

            # 1. Wait for communication
            bucket.comm_handle.wait()

            # 2. Average gradients
            bucket.flat_buffer.div_(self.world_size)

            # 3. Copy back to parameter gradients
            for p in bucket.params:
                if p.grad is not None:
                    p.grad.copy_(bucket.param_to_view[p])
