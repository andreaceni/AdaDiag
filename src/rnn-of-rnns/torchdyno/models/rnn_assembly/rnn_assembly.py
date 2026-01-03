# SET HERE eul_step GATE BEHAVIOUR
#[only_bias, input_and_state] = [False, True]
input_and_state = True

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

from .block_diagonal import BlockDiagonal
from .skew_symm_coupling import (
    SkewAntisymmetricCoupling,
    get_coupling_indices,
)



class RNNAssembly(nn.Module):
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
        gated_eul: bool = False,  # If True, use gated Euler step
        min_gate: float = 0.0001,
        max_gate: float = 0.1,
        coupling_rescaling: str = "local",  # "local", "global", or "unconstrained": controls spectral constraint on coupling blocks
        spectral_threshold: float = 1.0,  # New parameter for coupling spectral norm constraint
        structure: str = "identity",  # NEW: pass structure to BlockDiagonal
        diffusion: float = 0.0,  # Diffusion hyperparameter
    ):
        """Initializes the RNN of RNNs layer.

        Args:
            input_size (int): size of the input.
            out_size (int): size of the output.
            blocks (List[torch.Tensor]): list of blocks.
            coupling_blocks (List[torch.Tensor]): list of coupling blocks.
            coupling_topology (Union[int, float, List[Tuple[int, int]]]): coupling topology.
            eul_step (float, optional): Euler step. Defaults to 1e-2.
            gamma (Optional[float], optional): gamma parameter. Defaults to None.
            activation (str, optional): activation function. Defaults to "tanh".
            constrained_blocks (Optional[Literal["fixed", "tanh"]], optional):
                type of constraint. Defaults to None.
            dtype (torch.dtype, optional): data type. Defaults to torch.float32.
            gated_eul (bool, optional): If True, use gated Euler step.
            min_gate (float, optional): minimum step size for gating. Defaults to 0.0001
            max_gate (float, optional): maximum step size for gating. Defaults to 0.1
        """
        super().__init__()
        self._input_size = input_size
        self._gamma = gamma
        self._activation = activation
        self._dtype = dtype
        self.gated_eul = gated_eul
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.structure = structure

        if self.gated_eul:
            # make eul_step a learnable vector (of time scales) for each block
            self._hidden_size = sum(blocks[i].shape[0] for i in range(len(blocks)))

            # these lines are used only if gate is input-dependent and state-dependent
            self.gate_input = nn.Linear(input_size, self._hidden_size, dtype=dtype)
            if input_and_state:
                #self.gate_state = nn.Linear(self._hidden_size, self._hidden_size, bias=False) # dense h2h weights inside the gate
                self.gate_state = nn.Parameter(torch.zeros(self._hidden_size)) # diagonal h2h weights inside the gate
        else:
            self._eul_step = eul_step


        self._blocks = BlockDiagonal(
            blocks=blocks,
            constrained=constrained_blocks,
            gamma=gamma,
            dtype=dtype,
            structure=self.structure,
        )
        
        # Initialize V matrix for input-dependent diagonal constraints
        if constrained_blocks in ["tanh_input_state"]:
            self._blocks.initialize_hyperdiag_V(input_size)

        self._couplings = SkewAntisymmetricCoupling(
            block_sizes=self._blocks.block_sizes,
            coupling_blocks=coupling_blocks,
            coupling_topology=coupling_topology,
            dtype=self._dtype,
            coupling_rescaling=coupling_rescaling,
            spectral_threshold=spectral_threshold,
            diffusion=diffusion,
        )

        self._input_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, self._input_size),
                dtype=self._dtype,
            ),
            requires_grad=True,
        )

        self._out_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(out_size, self.hidden_size),
                dtype=self._dtype,
            ),
        )

        self.activ_fn = getattr(torch, self._activation)

    def compute_gate(self, xt, state):
        """Compute the gate for the gated Euler step."""

        # Ensure is 2D (these lines are used only if gate is input-dependent and state-dependent) 
        inpi = xt.unsqueeze(0) if xt.ndim == 1 else xt
        if input_and_state:
            stato = state.unsqueeze(0) if state.ndim == 1 else state
            #gate_logits = self.gate_input(inpi) + self.gate_state(stato) # state included in gate computation (dense h2h weights)
            gate_logits = self.gate_input(inpi) + self.gate_state * stato # state included in gate computation (diagonal h2h weights)
        else:
            gate_logits = self.gate_input(inpi) # state is not included in gate computation
        gate = torch.sigmoid(gate_logits)

        bounded_gate = self.min_gate + gate * (self.max_gate - self.min_gate)
        return bounded_gate

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_effective_trainable_parameters(self):
        """Counts all truly trainable parameters, including block and coupling modules."""
        total = 0
        if hasattr(self._blocks, "count_effective_trainable_parameters"):
            total += self._blocks.count_effective_trainable_parameters()
        else:
            for p in self._blocks.parameters():
                if p.requires_grad:
                    total += p.numel()
        if hasattr(self._couplings, "count_effective_trainable_parameters"):
            total += self._couplings.count_effective_trainable_parameters()
        else:
            for p in self._couplings.parameters():
                if p.requires_grad:
                    total += p.numel()
        # Add any other trainable parameters (e.g., input/output matrices, gates)
        for name, param in self.named_parameters():
            if (
                ("_blocks" not in name)
                and ("_couplings" not in name)
                and param.requires_grad
            ):
                total += param.numel()
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
        gated_eul: bool = False,  # If True, use gated Euler step
        min_gate: float = 0.0001,
        max_gate: float = 0.1,
        coupling_rescaling: str = "local",  # "local", "global", or "unconstrained": controls spectral constraint on coupling blocks
        spectral_threshold: float = 1.0,  # New parameter for coupling spectral norm constraint
        structure: str = "identity",  #     NEW: pass structure to BlockDiagonal
        diffusion: float = 0.0,  # Diffusion hyperparameter
    ) -> "RNNAssembly":
        """Create an RNNAssembly from initializers.

        Args:
            input_size (int): size of the input.
            out_size (int): size of the output.
            block_sizes (List[int]): list of block sizes.
            block_init_fn (Union[str, Callable[[torch.Size], torch.Tensor]]): block
                initializer.
            coupling_block_init_fn (Union[str, Callable[[torch.Size], torch.Tensor]]):
                coupling block initializer.
            coupling_topology (Union[int, float, List[Tuple[int, int]], Literal["ring"]]):
                coupling topology.
            eul_step (float, optional): Euler step. Defaults to 1e-2.
            activation (str, optional): activation function. Defaults to "tanh".
            constrained_blocks (Optional[Literal["fixed", "tanh"]], optional):
                type of constraint. Defaults to None.
            dtype (torch.dtype, optional): data type. Defaults to torch.float32.
            gated_eul (bool, optional): If True, use gated Euler step.
            min_gate (float, optional): minimum step size for gating. Defaults to 0.0001
            max_gate (float, optional): maximum step size for gating. Defaults to 0.1
        """

        if isinstance(block_init_fn, str):
            block_init_fn_: Callable = getattr(initializers, block_init_fn)
        else:
            block_init_fn_ = block_init_fn

        if isinstance(coupling_block_init_fn, str):
            coupling_block_init_fn_ = getattr(initializers, coupling_block_init_fn)
        else:
            coupling_block_init_fn_ = coupling_block_init_fn

        blocks = [block_init_fn_((b_size, b_size), dtype) for b_size in block_sizes]
        coupling_indices = get_coupling_indices(block_sizes, coupling_topology)
        coupling_blocks = [
            coupling_block_init_fn_((block_sizes[i], block_sizes[j]), dtype)
            for i, j in coupling_indices
        ]

        return RNNAssembly(
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
        )

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size, dtype=self._dtype).to(self._input_mat)
        input = input.to(dtype=self._dtype)

        states = self.compute_states(input, initial_state, mask)
        output = F.linear(states, self._out_mat)
        if torch.is_complex(output):
            output = torch.real(output)
        return output, states

    def compute_states(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            # Ensure all terms are on the same device as state
            # For hyperdiag_norm_dp1r, pass the current input to blocks
            if self._blocks._constrained == "tanh_input_state":
                blocks_out = self._blocks(self.activ_fn(state), current_input=xt, current_state=state)
            else:
                blocks_out = self._blocks(self.activ_fn(state))
            if isinstance(blocks_out, torch.Tensor):
                blocks_out = blocks_out.to(state.device)
            couplings_out = self._couplings(state).to(state.device)
            input_out = F.linear(xt, self._input_mat).to(state.device)

            if self.gated_eul:
                # Gated Euler step            
                gated_step = self.compute_gate(xt, state) 
                state = state + gated_step * ( # use gated euler step
                    -state
                    + blocks_out
                    + couplings_out
                    + input_out
                )
            else:
                state = state + self._eul_step * ( # use fixed euler step
                    -state
                    + blocks_out
                    + couplings_out
                    + input_out
                )
            if mask is not None:
                states.append(mask.to(device) * state)
            else:
                states.append(state)
        return torch.stack(states, dim=0)


    @property
    def hidden_size(self) -> int:
        return self._blocks.layer_size
