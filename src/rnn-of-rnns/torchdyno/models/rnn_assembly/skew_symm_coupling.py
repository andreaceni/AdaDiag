import random
from typing import (
    List,
    Literal,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torchdyno.models.initializers import block_diagonal_coupling


################################################################################################################
skew_couple = True # if True enforce skew-symmetry of the coupling matrix, otherwise random coupling matrix
#skew_couple = False # this is used only for Fig. 4 in the paper
################################################################################################################


class SkewAntisymmetricCoupling(nn.Module):

    def __init__(
        self,
        block_sizes: List[int],
        coupling_blocks: List[torch.Tensor],
        coupling_topology: List[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        coupling_rescaling: str = "local",  # "local", "global", or "unconstrained": controls spectral constraint on coupling blocks
        spectral_threshold: float = 1.0,  # New parameter for spectral constraint
        diffusion: float = 0.0,  # Diffusion hyperparameter 
    ):

        """
        Initializes the skew antisymmetric coupling layer.
        """

        super().__init__()
        self._block_sizes = block_sizes
        self._coupling_topology = coupling_topology
        self._dtype = dtype
        if len(coupling_blocks) != len(coupling_topology):
            raise ValueError(
                "The number of coupling blocks must be equal to the number of coupling topologies."
            )

        self.spectral_threshold = spectral_threshold
        self.coupling_rescaling = coupling_rescaling

        self._couplings = nn.Parameter(
            block_diagonal_coupling(
                block_sizes,
                [
                    (i, j, coupling_blocks[idx].to(dtype))
                    for idx, (i, j) in enumerate(coupling_topology)
                ],
                dtype=dtype,
            ),
        )
        base_mask = (self._couplings != 0)
        self._couple_mask = nn.Parameter(base_mask, requires_grad=False)
        self._cached_coupling = None

        # counting maximum number of incoming connections to any block
        self.max_incoming = self.max_incoming_connections()
        print(f"Maximum number of incoming connections to any block: {self.max_incoming}")

        if skew_couple:
            n_blocks = len(block_sizes)
            idx_i, idx_j = torch.triu_indices(n_blocks, n_blocks, offset=1)
            self.register_buffer("idx_i", idx_i)
            self.register_buffer("idx_j", idx_j)
        else:
            n_blocks = len(block_sizes)
            idx = torch.arange(n_blocks)
            row_idx, col_idx = torch.meshgrid(idx, idx, indexing="ij")
            off_diag_mask = row_idx != col_idx
            idx_i = row_idx[off_diag_mask]
            idx_j = col_idx[off_diag_mask]
            self.register_buffer("idx_i", idx_i)
            self.register_buffer("idx_j", idx_j)



    def max_incoming_connections(self) -> int:
        """Returns the maximum number of incoming connections to any block."""
        from collections import defaultdict

        incoming_counts = defaultdict(int)

        for i, j in self._coupling_topology:
            if i != j:
                incoming_counts[i] += 1  # i receives from j

        if not incoming_counts:
            return 0

        return max(incoming_counts.values())



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.couplings)

    def eval(self) -> None:
        super().eval()
        self._cached_coupling = None




    @property
    def couplings(self) -> torch.Tensor:
        if self._cached_coupling is None or self.training:
            couple_masked: torch.Tensor = self._couple_mask * self._couplings

            # TODO: THIS CAN BE IMPROVED TO SPEED UP TRAINING

            if skew_couple:
                anti_support = (
                    couple_masked.conj().T
                    if couple_masked.is_complex()
                    else couple_masked.T
                )
                self._cached_coupling = couple_masked - anti_support
            else:          
                self._cached_coupling = couple_masked

        return self._cached_coupling

        
    def count_effective_trainable_parameters(self):
        """
        Manually count the number of truly trainable coupling parameters based on topology and skew_couple.
        """
        # TODO: handle exact_row_couplings topology correctly
        block_size = self._block_sizes[0] if len(self._block_sizes) > 0 else 0
        n_couplings = len(self._coupling_topology)
        if skew_couple:
            # Each off-diagonal block is block_size x block_size, but only one direction is parameterized
            return n_couplings * block_size**2
        else:
            # TODO: handle the case when skew_couple is False, here we assume that both directions are parameterized
            # Each off-diagonal block is block_size x block_size, both directions parameterized
            return n_couplings * block_size**2 * 2


def get_coupling_indices(
    block_sizes: List[int],
    coupling_topology: Union[int, float, Literal["ring"], List[Tuple[int, int]], dict],
) -> List[Tuple[int, int]]:
    """Returns the coupling indices based on the topology.

    If coupling_topology is a dict with key "exact_row_couplings", its value is the number of nonzero off-diagonal blocks per block row.

    Args:
        block_sizes (List[int]): list of block sizes.
        coupling_topology (Union[int, float, Literal["ring"]]): coupling topology.

    Returns:
        List[Tuple[int, int]]: list of coupling indices.
    """

    if isinstance(coupling_topology, dict) and "exact_row_couplings" in coupling_topology:
        n_blocks = len(block_sizes)
        n_couplings = coupling_topology["exact_row_couplings"]
        if n_couplings >= n_blocks:
            raise ValueError("exact_row_couplings must be less than the number of blocks")
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for exact_row_couplings option")
        G = nx.random_regular_graph(n_couplings, n_blocks)
        pairs = set()
        for i, j in G.edges():
            pairs.add((i, j))
            pairs.add((j, i))
        coupling_indices = sorted(pairs)
    elif isinstance(coupling_topology, (int, float)):
        coupling_indices = [
            (i, j)
            for i in range(len(block_sizes) - 1)
            for j in range(i + 1, len(block_sizes))
        ]
        if coupling_topology > 0 and coupling_topology <= 1:
            coupling_topology = int(coupling_topology * len(coupling_indices))

        coupling_indices = random.sample(
            coupling_indices, int(min(len(coupling_indices), coupling_topology))
        )
    elif coupling_topology == "ring":
        coupling_indices = [
            (i, (i + 1) % len(block_sizes)) for i in range(len(block_sizes))
        ]
    else:
        coupling_indices = coupling_topology

    return coupling_indices
