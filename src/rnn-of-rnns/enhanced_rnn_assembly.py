"""
Enhanced RNNAssembly model with LayerNorm and attention-based pooling for Bayesian search experiments.
This file contains modifications specifically added for the Bayesian search and should not affect the existing codebase.
"""

from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import (
    Tensor,
    nn,
)
from torch.utils.data import DataLoader

from torchdyno.models import initializers

# Import from the local files in the current directory structure
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from torchdyno.models.rnn_assembly.block_diagonal import BlockDiagonal
    from torchdyno.models.rnn_assembly.skew_symm_coupling import (
        SkewAntisymmetricCoupling,
        get_coupling_indices,
    )
    from torchdyno.models.rnn_assembly import RNNAssembly
except ImportError:
    # Fallback for local development - try relative imports
    try:
        from .torchdyno.models.rnn_assembly.block_diagonal import BlockDiagonal
        from .torchdyno.models.rnn_assembly.skew_symm_coupling import (
            SkewAntisymmetricCoupling,
            get_coupling_indices,
        )
        from .torchdyno.models.rnn_assembly import RNNAssembly
    except ImportError:
        # Final fallback - assume local files
        from torchdyno.models.rnn_assembly.block_diagonal import BlockDiagonal
        from torchdyno.models.rnn_assembly.skew_symm_coupling import (
            SkewAntisymmetricCoupling,
            get_coupling_indices,
        )
        from torchdyno.models.rnn_assembly.rnn_assembly import RNNAssembly


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism for aggregating hidden states.
    Uses dot-product attention with learnable query vector.
    
    Added for Bayesian search experiments.
    """
    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        # Learnable query vector q shared across all time steps
        self.query = nn.Parameter(
            torch.normal(mean=0, std=1/np.sqrt(hidden_size), size=(hidden_size,), dtype=dtype),
            requires_grad=True
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to hidden states.
        
        Args:
            hidden_states: [seq_len, batch_size, hidden_size] or [seq_len, hidden_size]
            
        Returns:
            pooled_output: [batch_size, hidden_size] or [hidden_size]
        """
        # Compute attention scores: alpha_t = softmax(q^T h_t)
        # hidden_states: [seq_len, batch_size, hidden_size] or [seq_len, hidden_size]
        
        if hidden_states.dim() == 2:
            # Single sequence case: [seq_len, hidden_size]
            # scores: [seq_len]
            scores = torch.matmul(hidden_states, self.query)
            attention_weights = F.softmax(scores, dim=0)  # [seq_len]
            # Weighted sum: [hidden_size]
            pooled = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=0)
        else:
            # Batch case: [seq_len, batch_size, hidden_size]
            # scores: [seq_len, batch_size]
            scores = torch.matmul(hidden_states, self.query)
            attention_weights = F.softmax(scores, dim=0)  # [seq_len, batch_size]
            # Weighted sum: [batch_size, hidden_size]
            pooled = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=0)
            
        return pooled


class EnhancedRNNAssembly(RNNAssembly):
    """
    Enhanced RNNAssembly with LayerNorm stabilization and attention-based pooling.
    
    Modifications added for Bayesian search experiments:
    1. LayerNorm in the recurrent cell for better stability
    2. Attention-based pooling instead of just using the last hidden state
    
    This class inherits from the original RNNAssembly to avoid breaking existing functionality.
    """
    
    def __init__(
        self,
        input_size: int,
        out_size: int,
        blocks: List[torch.Tensor],
        coupling_blocks: List[torch.Tensor],
        coupling_topology: List[Tuple[int, int]],
        eul_step: float = 1e-2,
        gamma: Optional[float] = None,
        activation: str = "tanh",
        constrained_blocks: Optional[
            Literal["fixed", "tanh"]
        ] = None,
        dtype: torch.dtype = torch.float32,
        gated_eul: bool = False,
        min_gate: float = 0.0001,
        max_gate: float = 0.1,
        coupling_rescaling: str = "local",
        spectral_threshold: float = 1.0,
        structure: str = "identity",
        diffusion: float = 0.0,
        # New parameters for Bayesian search enhancements
        use_layer_norm: bool = True,  # Added for Bayesian search
        use_attention_pooling: bool = True,  # Added for Bayesian search
    ):
        """Enhanced RNNAssembly with LayerNorm and attention pooling."""
        
        # Initialize the parent class
        super().__init__(
            input_size=input_size,
            out_size=out_size,
            blocks=blocks,
            coupling_blocks=coupling_blocks,
            coupling_topology=coupling_topology,
            eul_step=eul_step,
            gamma=gamma,
            activation=activation,
            constrained_blocks=constrained_blocks,
            dtype=dtype,
            gated_eul=gated_eul,
            min_gate=min_gate,
            max_gate=max_gate,
            coupling_rescaling=coupling_rescaling,
            spectral_threshold=spectral_threshold,
            structure=structure,
            diffusion=diffusion,
        )
        
        # Added for Bayesian search: LayerNorm for stabilization
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size, dtype=dtype)
        
        # Added for Bayesian search: Attention pooling mechanism
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            self.attention_pooling = AttentionPooling(self.hidden_size, dtype=dtype)

    def count_effective_trainable_parameters(self):
        """
        Counts all truly trainable parameters, including the new components added for Bayesian search.
        """
        # Get base parameters from parent class
        total = super().count_effective_trainable_parameters()
        
        # Added for Bayesian search: Count LayerNorm parameters
        if self.use_layer_norm:
            total += sum(p.numel() for p in self.layer_norm.parameters() if p.requires_grad)
        
        # Added for Bayesian search: Count attention pooling parameters
        if self.use_attention_pooling:
            total += sum(p.numel() for p in self.attention_pooling.parameters() if p.requires_grad)
        
        return total

    @staticmethod
    def from_initializers(
        input_size: int,
        out_size: int,
        block_sizes: List[int],
        block_init_fn: Union[str, Callable[[torch.Size], torch.Tensor]],
        coupling_block_init_fn: Union[str, Callable[[torch.Size], torch.Tensor]],
        coupling_topology: Union[int, float, List[Tuple[int, int]], Literal["ring"]],
        eul_step: float = 1e-2,
        gamma: Optional[float] = None,
        activation: str = "tanh",
        constrained_blocks: Optional[
            Literal["fixed", "tanh"]
        ] = None,
        dtype: torch.dtype = torch.float32,
        gated_eul: bool = False,
        min_gate: float = 0.0001,
        max_gate: float = 0.1,
        coupling_rescaling: str = "local",
        spectral_threshold: float = 1.0,
        structure: str = "identity",
        diffusion: float = 0.0,
        # New parameters for Bayesian search enhancements
        use_layer_norm: bool = True,  # Added for Bayesian search
        use_attention_pooling: bool = True,  # Added for Bayesian search
    ) -> "EnhancedRNNAssembly":
        """Create an EnhancedRNNAssembly from initializers."""

        # Added for Bayesian search: Use fixed initializer functions
        def get_init_fn_safe(fn_str: str, *args):
            """Safe version of get_init_fn that handles uniform correctly."""
            from torchdyno.models.initializers import sparse, diagonal, orthogonal, zeros, uniform
            import torch

            if fn_str == "sparse":
                return lambda x, dtype: sparse(x, *args, dtype=dtype)
            if fn_str == "orthogonal":
                return lambda x, dtype: orthogonal(x, dtype=dtype)
            if fn_str == "diagonal":
                return lambda x, dtype: diagonal(x, dtype=dtype)
            if fn_str == "lt":
                return lambda x, dtype: diagonal(x, dtype=dtype) + torch.tril(uniform(x, dtype=dtype))
            if fn_str == "uniform":
                return lambda x, dtype: uniform(x, dtype=dtype)  # Fixed: removed None, None arguments
            if fn_str == "zeros":
                return lambda x, dtype: zeros(x, dtype=dtype)
            raise ValueError(f"Unknown initialization function {fn_str}")

        # Handle initializer functions properly
        if isinstance(block_init_fn, str):
            block_init_fn_ = get_init_fn_safe(block_init_fn)
        else:
            block_init_fn_ = block_init_fn

        if isinstance(coupling_block_init_fn, str):
            coupling_block_init_fn_ = get_init_fn_safe(coupling_block_init_fn)
        else:
            coupling_block_init_fn_ = coupling_block_init_fn

        # Create blocks with proper function signature
        blocks = [block_init_fn_((b_size, b_size), dtype) for b_size in block_sizes]
        coupling_indices = get_coupling_indices(block_sizes, coupling_topology)
        coupling_blocks = [
            coupling_block_init_fn_((block_sizes[i], block_sizes[j]), dtype)
            for i, j in coupling_indices
        ]

        return EnhancedRNNAssembly(
            input_size=input_size,
            out_size=out_size,
            blocks=blocks,
            coupling_blocks=coupling_blocks,
            coupling_topology=coupling_indices,
            eul_step=eul_step,
            gamma=gamma,
            activation=activation,
            constrained_blocks=constrained_blocks,
            dtype=dtype,
            gated_eul=gated_eul,
            min_gate=min_gate,
            max_gate=max_gate,
            coupling_rescaling=coupling_rescaling,
            spectral_threshold=spectral_threshold,
            structure=structure,
            diffusion=diffusion,
            use_layer_norm=use_layer_norm,  # Added for Bayesian search
            use_attention_pooling=use_attention_pooling,  # Added for Bayesian search
        )

    def compute_states(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enhanced state computation with LayerNorm stabilization.
        Added for Bayesian search experiments.
        """
        states = []
        device = self._input_mat.device
        state = (
            initial_state.to(device)
            if initial_state is not None
            else torch.zeros(self.hidden_size, dtype=self._dtype, device=device)
        )
        timesteps = input.shape[0]
        for t in range(timesteps):
            # Ensure input[t] is on the correct device
            xt = input[t].to(device)
            
            # For hyperdiag_norm_dp1r, pass the current input to blocks
            if self._blocks._constrained == "tanh_input_state":
                blocks_out = self._blocks(self.activ_fn(state), current_input=xt, current_state=state)
            else:
                blocks_out = self._blocks(self.activ_fn(state))
            if isinstance(blocks_out, torch.Tensor):
                blocks_out = blocks_out.to(state.device)
            couplings_out = self._couplings(state).to(state.device)
            input_out = F.linear(xt, self._input_mat).to(state.device)

            # Compute the state update
            state_update = -state + blocks_out + couplings_out + input_out

            if self.gated_eul:
                # Gated Euler step            
                gated_step = self.compute_gate(xt, state) 
                state = state + gated_step * state_update
            else:
                state = state + self._eul_step * state_update
            
            # Added for Bayesian search: Apply LayerNorm for stabilization
            if self.use_layer_norm:
                state = self.layer_norm(state)
                
            if mask is not None:
                states.append(mask.to(device) * state)
            else:
                states.append(state)
        return torch.stack(states, dim=0)

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enhanced forward pass with attention-based pooling.
        Added for Bayesian search experiments.
        """
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size, dtype=self._dtype).to(self._input_mat)
        input = input.to(dtype=self._dtype)

        states = self.compute_states(input, initial_state, mask)
        
        # Added for Bayesian search: Use attention pooling instead of just the last state
        if self.use_attention_pooling:
            # Apply attention pooling to all hidden states
            pooled_state = self.attention_pooling(states)  # [batch_size, hidden_size] or [hidden_size]
            
            # Ensure proper dimensions for output layer
            if pooled_state.dim() == 1:
                # Single sequence: [hidden_size] -> [1, hidden_size] for linear layer
                pooled_state = pooled_state.unsqueeze(0)
            
            # Apply output layer to get final predictions for each sample in the batch
            output = F.linear(pooled_state, self._out_mat)  # [batch_size, out_size]
        else:
            # Original behavior: use only the last state
            output = F.linear(states, self._out_mat)
            
        if torch.is_complex(output):
            output = torch.real(output)
        return output, states