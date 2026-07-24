from typing import Any, Type, Iterable, Dict, List
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    Sharded optimizer (Optimizer State Sharding, simplified ZeRO-1)

    Core idea:
    - All ranks share the full model parameters (parameters themselves are not sharded)
    - But each rank is "responsible" for the optimizer state of only a subset of parameters
      (e.g., Adam's m / v)
    - Each step:
        1. Each rank runs optimizer.step() only on the parameters it owns
        2. Then broadcast the updated parameters to all ranks

    This allows:
    - Reducing optimizer-state memory footprint to 1 / world_size of the original
    - Being logically equivalent to single-device training (after synchronization)
    """

    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
        # ZeRO / sharded optimizer must run after torch.distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")

        # Current process rank (globally unique)
        self.rank = dist.get_rank()

        # Total number of processes (world size)
        self.world_size = dist.get_world_size()

        # Wrapped real optimizer type (e.g., torch.optim.Adam)
        self.optimizer_cls = optimizer_cls

        # Hyperparameters for the real optimizer (lr, betas, etc.)
        self.optimizer_kwargs = kwargs

        # Call the Optimizer constructor
        # Effects:
        # - Initialize self.param_groups
        # - Call add_param_group for each parameter group
        super().__init__(params, defaults={})

        # Build the local optimizer specific to this rank
        self._build_local_optimizer()

    def _build_local_optimizer(self):
        """
        Build the local optimizer for the current rank.

        Core logic:
        - Iterate over all parameters
        - Keep only parameters whose owner is the current rank
        - Create the actual optimizer that executes step() on these parameters

        Note:
        - Each rank's local_optimizer sees a different subset of parameters
        - But the param_groups structure (lr / weight_decay, etc.) stays consistent
        """
        local_param_groups: List[Dict[str, Any]] = []

        for group in self.param_groups:
            # From the current parameter group, keep parameters whose owner is the current rank
            local_params = [
                p for p in group["params"]
                if self._param_owner(p) == self.rank
            ]

            # Skip this group if the current rank has no parameters in it
            if len(local_params) == 0:
                continue

            # Copy the group config (avoid modifying the original param_groups)
            local_group = dict(group)

            # Replace params with those owned by this rank
            local_group["params"] = local_params

            local_param_groups.append(local_group)

        # Instantiate the real optimizer (e.g., Adam) with the filtered parameter groups
        self.local_optimizer = self.optimizer_cls(
            local_param_groups,
            **self.optimizer_kwargs,
        )

    def _param_owner(self, param: torch.nn.Parameter) -> int:
        """
        Determine which rank "owns" a parameter.

        Current strategy:
        - Assign a globally unique index to each parameter
        - owner = index % world_size

        This is the simplest and most common static sharding scheme.
        """
        return self._global_param_index(param) % self.world_size

    def _global_param_index(self, param: torch.nn.Parameter) -> int:
        """
        Assign a stable "global index" to every parameter.

        Requirements:
        - Parameter traversal order must be the same on all ranks
        - The same parameter must have the exact same index on different ranks

        Implementation:
        - On first call, iterate over all param_groups
        - Number each parameter in order of appearance
        - Cache the param -> index mapping in a dict
        """
        if not hasattr(self, "_param_to_index"):
            self._param_to_index = {}
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    self._param_to_index[p] = idx
                    idx += 1
        return self._param_to_index[param]

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        Execute one optimization step (equivalent to Optimizer.step).

        Flow:
        1. (Optional) Run closure to compute loss
        2. Run optimizer.step() only on parameters owned by the current rank
        3. Broadcast updated parameters to all ranks to synchronize model parameters

        Note:
        - Optimizer state (e.g., Adam momentum) exists only on the owner rank
        - Parameters are consistent across all ranks after step()
        """
        loss = None
        if closure is not None:
            # closure needs gradients enabled
            with torch.enable_grad():
                loss = closure()

        # Only update parameters owned by this rank
        self.local_optimizer.step(**kwargs)

        # Synchronize parameters: owner rank -> all other ranks
        self._sync_parameters()

        return loss

    def _sync_parameters(self):
        """
        Parameter synchronization logic (broadcast).

        For each parameter:
        - Find its owner rank
        - Broadcast the parameter from the owner rank to all ranks

        This guarantees:
        - Although each parameter is updated on only one rank
        - After step(), parameter values are identical on all ranks
        """
        for group in self.param_groups:
            for p in group["params"]:
                owner = self._param_owner(p)
                dist.broadcast(p.data, src=owner)

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Dynamically add a parameter group (consistent with PyTorch Optimizer interface).

        Because parameters have changed:
        - Recompute parameter sharding
        - Rebuild the local optimizer
        """
        super().add_param_group(param_group)
        # Invalidate the cached param->index map so it's rebuilt against the
        # full (now-larger) set of param_groups, including the new params.
        if hasattr(self, "_param_to_index"):
            del self._param_to_index
        self._build_local_optimizer()
