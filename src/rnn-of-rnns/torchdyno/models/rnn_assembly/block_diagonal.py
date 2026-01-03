import warnings

def module_structure(diag_blocks, structure="identity"):
    """
    TODO: not all configurations employs the module_structure, e.g. idx=3 cannot use it, I guess.

    Returns a batch of block matrices according to the specified structure.
    - diag_blocks: shape [n_blocks, block_size]
    - structure: "identity" or "ring"
    """
    if structure == "identity":
        return torch.diag_embed(diag_blocks)
    elif structure == "ring":
        # Each block: roll identity by 1, multiply by diag_blocks as column vector
        n_blocks, block_size = diag_blocks.shape
        eye = torch.eye(block_size, device=diag_blocks.device, dtype=diag_blocks.dtype)
        perm = torch.roll(eye, shifts=1, dims=1)  # shape [block_size, block_size]
        # Broadcast diag_blocks to [n_blocks, block_size, block_size]
        return diag_blocks.unsqueeze(-1) * perm.unsqueeze(0)
    else:
        warnings.warn(f"Unknown structure '{structure}', defaulting to identity.")
        return torch.diag_embed(diag_blocks)
    
    
from typing import (
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.models.initializers import block_diagonal


class BlockDiagonal(nn.Module):
    """
    Extended to support "proper_diag_norm_dp1r" constraint for block_config=9:
    - Diagonal entries constrained to abs < 1/2
    - Two vectors (for rank-1 part) constrained to norm < 1/sqrt(2)
    """

    def __init__(
        self,
        blocks: List[torch.Tensor],
        bias: bool = False,
        constrained: Optional[Literal["fixed", "tanh"]] = None,
        gamma: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        structure: str = "ring", # define the structure of the single RNN modules
    ):
        """Initializes the block diagonal matrix.

        Args:
            blocks (List[torch.Tensor]): list of blocks.
            bias (bool, optional): whether to use bias. Defaults to False.
            constrained (Optional[Literal["fixed", "tanh"]], optional):
                type of constraint. Defaults to None.
        """
        super().__init__()
        self._block_sizes = [block.size(0) for block in blocks]
        self._constrained = constrained
        self._gamma = gamma
        self._dtype = dtype
        self._structure = structure

        # Special handling for "tanh_input_state"
        if constrained == "tanh_input_state":
            # Input+state-dependent diagonal WITHOUT rank-1: real-valued only
            if dtype.is_complex:
                raise ValueError("tanh_input_state constraint only supports real-valued tensors")
            block_size = blocks[0].size(0)
            n_blocks = len(blocks)
            layer_size = sum(self._block_sizes)
                                
            # Parameters for input+state-dependent diagonal: V matrix, W matrices, and bias
            # Diagonal gate: tanh(V * xt + W_i * h_{i,t} + b) (NO 0.5 multiplier)
            # V should transform input_size -> layer_size
            # W_i should transform block_size -> block_size for each block i (diagonal)
            self._hyperdiag_V = None  # Will be initialized when input_size is known
            # Use diagonal h2h matrices: shape [n_blocks, block_size] (same as idx=12)
            self._tanhinputstate_W = nn.Parameter(torch.empty(n_blocks, block_size).uniform_(-0.1, 0.1))
            self._hyperdiag_b = nn.Parameter(torch.zeros(layer_size))
                
            self._tanhinputstate_block_size = block_size
            self._tanhinputstate_n_blocks = n_blocks
            self._blocks = None  # Not used
            self._input_size = None  # Will be set during initialization

        else:
            self._blocks = nn.Parameter(
                block_diagonal(blocks, dtype=dtype),
                requires_grad=constrained != "fixed",
            )
            self._blocks_mask = nn.Parameter(
                (self._blocks != 0) if self._blocks is not None else torch.ones(sum([b.size(0) for b in blocks]), sum([b.size(0) for b in blocks]), dtype=torch.bool),
                requires_grad=False
            )
        self._support_eye = torch.eye(self.layer_size, dtype=dtype)
        self._eye_gamma = self._support_eye * self._gamma if self._gamma else None

        if bias:
            self.bias = nn.Parameter(
                torch.normal(
                    mean=0, std=(1 / np.sqrt(self.layer_size)), dtype=self._blocks.dtype if self._blocks is not None else dtype
                ),
            )
        else:
            self.bias = None
        self._cached_blocks = None
    

    def initialize_hyperdiag_V(self, input_size: int):
        """Initialize the V matrix for all input-dependent diagonal constraints when input size is known."""
        if self._constrained in ["tanh_input_state"] and self._hyperdiag_V is None:
            layer_size = sum(self._block_sizes)
            self._hyperdiag_V = nn.Parameter(
                torch.normal(mean=0, std=1/np.sqrt(input_size), size=(input_size, layer_size))
            )
            self._input_size = input_size

    def forward(self, x: torch.Tensor, current_input: torch.Tensor = None, current_state: torch.Tensor = None) -> torch.Tensor:
        # For tanh_input_state, we need both current input and current state
        if self._constrained == "tanh_input_state":
            if current_input is None:
                raise ValueError("tanh_input_state constraint requires current_input parameter")
            if current_state is None:
                raise ValueError("tanh_input_state constraint requires current_state parameter")
            blocks = self.get_blocks_with_input_and_state(current_input, current_state)
        else:
            blocks = self.blocks

        # Ensure input and weights are the same dtype
        x = x.to(blocks.dtype)
        if callable(blocks):
            return blocks(x)
        output = F.linear(x, blocks, self.bias)
        if output.is_complex() and output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

    def eval(self) -> None:
        super().eval()
        self._cached_blocks = None
        # Clear rank-1 component caches when switching to eval mode
        if hasattr(self, '_cached_v_c'):
            self._cached_v_c = None
        if hasattr(self, '_cached_w_c'):
            self._cached_w_c = None

    @property
    def n_blocks(self) -> int:
        return len(self._block_sizes)

    @property
    def block_sizes(self) -> List[int]:
        return self._block_sizes

    @property
    def layer_size(self) -> int:
        return sum(self._block_sizes)

    def count_effective_trainable_parameters(self):
        """
        Counts all truly trainable parameters, including off-diagonal blocks.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self._constrained == "fixed":
            # No trainable parameters in fixed case
            total = 0
        elif self._constrained == "tanh_input_state":
            # tanh_input_state has: V matrix + W matrices + b vector (NO rank-1)
            n_blocks = len(self._block_sizes)
            block_size = self._block_sizes[0] if n_blocks > 0 else 0
            layer_size = sum(self._block_sizes)
                        
            # Input+state-dependent diagonal: V matrix (input_size × layer_size) + W matrices (n_blocks × block_size) + bias (layer_size)
            if hasattr(self, '_hyperdiag_V') and self._hyperdiag_V is not None:
                input_size = self._hyperdiag_V.shape[0]  # V is (input_size, layer_size)
                hyperdiag_params = input_size * layer_size + layer_size
            else:
                # If V not initialized yet, we can't count its parameters
                hyperdiag_params = layer_size  # Just bias for now
            
            # W matrices: n_blocks × block_size (diagonal h2h)
            state_params = n_blocks * block_size
            
            total = hyperdiag_params + state_params
        else:
            # all other cases have just a diagonal matrix parameterization
            n_blocks = len(self._block_sizes)
            block_size = self._block_sizes[0] if n_blocks > 0 else 0
            total = n_blocks * block_size
        return total

    @property
    def blocks(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self._cached_blocks is None or (
            self.training and self._constrained != "fixed"
        ):
            if self._blocks is not None and self._blocks_mask is not None:
                blocks_ = self._blocks * self._blocks_mask
                if self._gamma:
                    if self._eye_gamma.device != blocks_.device:
                        self._eye_gamma = self._eye_gamma.to(blocks_.device)
                    blocks_ = blocks_ * self._eye_gamma

            if self._constrained == "tanh":
                blocks_ = torch.tanh(blocks_)

            self._cached_blocks = blocks_
                
        return self._cached_blocks


    def get_blocks_with_input_and_state(self, current_input: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """Compute blocks for tanh_input_state constraint with input+state-dependent diagonal."""
        if self._constrained != "tanh_input_state":
            return self.blocks
            
        # Check if V matrix is initialized
        if self._hyperdiag_V is None:
            raise ValueError("V matrix not initialized. Call initialize_hyperdiag_V() first.")
        
        # Prepare input: should be [input_size] or [batch_size, input_size]
        if current_input.dim() == 1:
            current_input = current_input.unsqueeze(0)  # Add batch dimension
        
        # Prepare state: should be [layer_size] or [batch_size, layer_size]
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)  # Add batch dimension
            
        batch_size = current_input.shape[0]
        
        # Handle batch size mismatch - expand state if needed
        if current_state.shape[0] != batch_size:
            if current_state.shape[0] == 1:
                current_state = current_state.expand(batch_size, -1)
            else:
                raise ValueError(f"Batch size mismatch: input has {batch_size}, state has {current_state.shape[0]}")
        
        # Step 1: Compute input-dependent part: V * xt
        input_contribution = F.linear(current_input, self._hyperdiag_V.T, None)  # [batch_size, layer_size]
        
        # Step 2: Compute state-dependent part per block: W_i * h_{i,t}
        n_blocks = self._tanhinputstate_n_blocks
        block_size = self._tanhinputstate_block_size
        
        # Split current state into blocks: [batch_size, n_blocks, block_size]
        state_blocks = current_state.reshape(batch_size, n_blocks, block_size)
        
        # Diagonal h2h: elementwise multiplication per block
        # state_blocks: [batch_size, n_blocks, block_size]
        # _tanhinputstate_W: [n_blocks, block_size]
        state_contribution_blocks = state_blocks * self._tanhinputstate_W.unsqueeze(0)
        
        # Reshape back to layer format: [batch_size, layer_size]
        state_contribution = state_contribution_blocks.reshape(batch_size, -1)
        
        # Step 3: Combine contributions and add bias
        # Diagonal gate: tanh(V * xt + W_i * h_{i,t} + b)
        diag_logits = input_contribution + state_contribution + self._hyperdiag_b.unsqueeze(0)
        diag_values = torch.tanh(diag_logits)  
        
        # If batch_size=1, squeeze to get [layer_size]
        if diag_values.shape[0] == 1:
            diag_values = diag_values.squeeze(0)
        
        # Create block-wise diagonal matrices from the input+state-dependent values
        # Handle batch dimension: diag_values can be [batch_size, layer_size] or [layer_size]
        if diag_values.dim() == 2:
            # Batch mode: [batch_size, layer_size] -> [batch_size, n_blocks, block_size]
            current_batch_size = diag_values.shape[0]
            diag_blocks = diag_values.reshape(current_batch_size, n_blocks, block_size)
            # For now, just use the first sample in the batch (this is a simplification)
            # In a proper implementation, we'd need to handle full batch processing
            diag_blocks = diag_blocks[0]  # [n_blocks, block_size]
        else:
            # Single sample: [layer_size] -> [n_blocks, block_size]
            diag_blocks = diag_values.reshape(n_blocks, block_size)
        
        # Diagonal matrices: [n_blocks, block_size, block_size]
        D = module_structure(diag_blocks, structure=self._structure)
        
        # NO rank-1 matrices for this constraint!
        
        # Block matrices: [n_blocks, block_size, block_size] (only diagonal)
        block_mats = D
        
        # Assemble block diagonal matrix
        blocks_ = torch.block_diag(*block_mats)
        
        return blocks_