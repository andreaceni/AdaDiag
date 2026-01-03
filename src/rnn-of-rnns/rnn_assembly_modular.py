import torch
from torch import nn
from torchdyno.models.rnn_assembly import RNNAssembly

class RNNAssemblyModular(RNNAssembly):
    """
    RNNAssembly variant where input channels are randomly assigned to modules (modules >= channels).
    Each input channel is assigned to at least one module, and modules may process multiple channels.
    This implementation distributes input channels randomly and leverages coupling between modules to learn the task.

    Key differences from base RNNAssembly:
    - Each module processes its assigned input channels (random assignment, all channels covered)
    - Coupling matrix allows information flow between modules
    - Number of modules must be greater than or equal to number of input channels
    """
    
    @staticmethod
    def from_initializers(
        input_size,
        out_size,
        block_sizes,
        block_init_fn,
        coupling_block_init_fn,
        coupling_topology,
        eul_step=1e-2,
        gamma=None,
        activation="tanh",
        constrained_blocks=None,
        dtype=torch.float32,
        gated_eul=False,
        min_gate=0.0001,
        max_gate=0.1,
        coupling_rescaling="local",
        spectral_threshold=1.0,
        structure="identity",
        diffusion=0.0,  # Pass diffusion parameter
    ):
        """
        Factory method to create RNNAssemblyModular with proper initialization.
        
        Args:
            input_size (int): Number of input channels (must equal number of modules)
            out_size (int): Number of output classes
            block_sizes (List[int]): Size of each RNN module 
            ... (other args same as RNNAssembly)
        """
        from torchdyno.models import initializers
        from torchdyno.models.rnn_assembly.rnn_assembly import get_coupling_indices

        if isinstance(block_init_fn, str):
            block_init_fn_ = getattr(initializers, block_init_fn)
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
        
        return RNNAssemblyModular(
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

    def __init__(self, *args, **kwargs):
        """Initialize the modular RNN assembly."""
        self.diffusion = kwargs.get("diffusion", 0.0)
        super().__init__(*args, **kwargs)
        
        # Validate that number of modules >= number of input channels
        num_modules = len(self._blocks.block_sizes)
        if self._input_size > num_modules:
            raise ValueError(f"Number of modules ({num_modules}) must be >= number of input channels ({self._input_size}) for modular structure.")
        
        # Random assignment: each input channel assigned to a module, all channels covered
        import random
        seed = kwargs.get("modular_seed", None)
        if seed is not None:
            random.seed(seed)
        self._modular_channel_assignment = self._assign_channels_randomly(self._input_size, num_modules)
        # Example: {0: [2], 1: [0,3], 2: [1]} means module 0 gets channel 2, module 1 gets channels 0 and 3, etc.

    @staticmethod
    def _assign_channels_randomly(n_channels, n_modules):
        """
        Randomly assign each input channel to a module, all channels covered.
        Returns: dict {module_idx: [channel_idx, ...]}
        """
        import random
        # Start with empty assignments
        assignments = {i: [] for i in range(n_modules)}
        # Shuffle channels and assign each to a random module
        channels = list(range(n_channels))
        random.shuffle(channels)
        for ch in channels:
            m = random.randint(0, n_modules-1)
            assignments[m].append(ch)
        # Ensure all channels are assigned (should always be true)
        assigned = [ch for lst in assignments.values() for ch in lst]
        assert sorted(assigned) == list(range(n_channels)), "Not all channels assigned!"
        return assignments

    def compute_states(
        self,
        input: torch.Tensor,
        initial_state: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute states with modular input distribution.
        
        Args:
            input: (seq_len, batch_size, input_size) tensor
            initial_state: Optional initial hidden state
            mask: Optional mask tensor
            
        Returns:
            states: (seq_len, batch_size, hidden_size) tensor
        """
        states = []
        device = self._input_mat.device
        
        # Initialize state
        if initial_state is None:
            state = torch.zeros(self.hidden_size, dtype=self._dtype, device=device)
        else:
            state = initial_state.to(device)
            
        seq_len, batch_size, input_size = input.shape
        num_modules = len(self._blocks.block_sizes)
        
        # If input_size matches num_modules, use modular distribution
        # Modular assignment: use random mapping
        if input_size <= num_modules:
            for t in range(seq_len):
                xt = input[t].to(device)  # (batch_size, input_size)
                state = self._modular_step(xt, state)
                if mask is not None:
                    states.append(mask.to(device) * state)
                else:
                    states.append(state)
        else:
            print(f"Using standard input distribution (input_size={input_size}, num_modules={num_modules})")
            return super().compute_states(input, initial_state, mask)
        return torch.stack(states, dim=0)

    def _modular_step(self, xt: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Perform one modular forward step.
        
        Args:
            xt: (batch_size, input_size) current input
            state: (batch_size, hidden_size) current state
            
        Returns:
            new_state: (batch_size, hidden_size) updated state
        """
        batch_size, input_size = xt.shape
        num_modules = len(self._blocks.block_sizes)
        
        # Create modular input matrix: each module gets its assigned input channels
        modular_input = torch.zeros(batch_size, self.hidden_size, device=state.device, dtype=self._dtype)
        start_idx = 0
        for module_idx in range(num_modules):
            block_size = self._blocks.block_sizes[module_idx]
            end_idx = start_idx + block_size
            # Get assigned channels for this module
            assigned_channels = self._modular_channel_assignment[module_idx]
            if assigned_channels:
                # Sum projections for all assigned channels
                channel_sum = torch.zeros(batch_size, block_size, device=state.device, dtype=self._dtype)
                for ch in assigned_channels:
                    channel_input = xt[:, ch].unsqueeze(1)  # (batch_size, 1)
                    input_projection = self._input_mat[start_idx:end_idx, ch:ch+1]  # (block_size, 1)
                    channel_sum += torch.mm(channel_input, input_projection.T)
                modular_input[:, start_idx:end_idx] = channel_sum
            start_idx = end_idx

        # Process through blocks with activation
        if self._blocks._constrained in ["hyperdiag_norm_dp1r", "gatehyperdiag_norm_dp1r", "tanh_input_state", "tanh_input"]:
            # For input-dependent constraints, pass the original input
            if self._blocks._constrained in ["gatehyperdiag_norm_dp1r", "tanh_input_state"]:
                blocks_out = self._blocks(self.activ_fn(state), current_input=xt, current_state=state)
            else:
                blocks_out = self._blocks(self.activ_fn(state), current_input=xt)
        else:
            blocks_out = self._blocks(self.activ_fn(state))
            
        if isinstance(blocks_out, torch.Tensor):
            blocks_out = blocks_out.to(state.device)

        # Coupling between modules (this is key for modular learning)
        couplings_out = self._couplings(state).to(state.device)

        # Update state using Euler integration
        if self.gated_eul:
            # Gated Euler step
            gated_step = self.compute_gate(xt, state)
            new_state = state + gated_step * (
                -state + blocks_out + couplings_out + modular_input
            )
        else:
            # Fixed Euler step
            new_state = state + self._eul_step * (
                -state + blocks_out + couplings_out + modular_input
            )
            
        return new_state

    def get_modular_info(self) -> dict:
        """
        Get information about the modular structure.
        
        Returns:
            dict: Information about input distribution and module structure
        """
        num_modules = len(self._blocks.block_sizes)
        return {
            "input_size": self._input_size,
            "num_modules": num_modules,
            "block_sizes": self._blocks.block_sizes,
            "is_modular": self._input_size == num_modules,
            "coupling_topology": len(self._couplings.coupling_topology) if hasattr(self._couplings, 'coupling_topology') else 0,
        }