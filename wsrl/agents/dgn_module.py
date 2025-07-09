"""
Standalone Data-Guided Noise (DGN) module for exploration in RL.
Can be composed with any RL agent to guide exploration using demonstration data.
"""
from typing import Optional, Tuple, Callable

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict
from functools import partial

from wsrl.common.common import JaxRLTrainState, ModuleDict
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.networks.mlp import MLP


class CovarianceNetwork(nn.Module):
    """Network that outputs multivariate normal distribution with learned covariance."""
    hidden_dims: Tuple[int, ...] = (256, 256)
    output_dim: int = None  # action_dim
    dropout_rate: float = 0.5
    cov_diagonal_eps: float = 1e-5
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True) -> distrax.Distribution:
        # Ensure observations has batch dimension
        if observations.ndim == 1:
            observations = observations[None, :]
            
        batch_size = observations.shape[0]
        action_dim = self.output_dim
        
        # Use MLP for hidden layers
        x = MLP(
            hidden_dims=self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate if train else 0.0,
        )(observations, train=train)
        
        # Output layer - outputs flat vector of lower triangular matrix elements
        num_params = action_dim * (action_dim + 1) // 2
        cholesky_elements = nn.Dense(num_params)(x)
        
        # Handle potential numerical issues
        cholesky_elements = jnp.nan_to_num(cholesky_elements, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Construct lower triangular matrix for each batch element
        tril_indices = jnp.tril_indices(action_dim)
        L = jnp.zeros((batch_size, action_dim, action_dim))
        L = L.at[:, tril_indices[0], tril_indices[1]].set(cholesky_elements)
        
        # Apply softplus to diagonal to ensure positive definiteness
        diag_indices = jnp.arange(action_dim)
        L = L.at[:, diag_indices, diag_indices].set(
            nn.softplus(L[:, diag_indices, diag_indices]) + self.cov_diagonal_eps
        )
        
        # Create zero-mean multivariate normal distribution
        zero_mean = jnp.zeros((batch_size, action_dim))
        dist = distrax.MultivariateNormalTri(
            loc=zero_mean,
            scale_tri=L
        )
        
        return dist


class DGNModule:
    
    def __init__(self, state: JaxRLTrainState, config: ConfigDict, demo_dataset: dict):
        self.state = state
        self.config = config
        self.demo_dataset = demo_dataset
        self.step = 0
    
    def replace(self, **kwargs):
        """Create a new instance with updated attributes."""
        new_module = self.__class__(
            state=kwargs.get("state", self.state),
            config=kwargs.get("config", self.config),
            demo_dataset=kwargs.get("demo_dataset", self.demo_dataset),
        )
        new_module.step = kwargs.get("step", self.step)
        return new_module
    
    def forward_covariance(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """Forward pass through covariance network."""
        return self.state.apply_fn(
            grad_params or self.state.params,
            observations,
            train=train,
            rngs={"dropout": rng} if train and rng is not None else {},
        )
    
    def sample_noise(self, observations: Data, rng: PRNGKey, total_env_steps: Optional[int] = None) -> jnp.ndarray:
        """Sample exploration noise from learned covariance with optional annealing."""
        # Check if single observation
        single_obs = observations.ndim == 1
        
        dist = self.forward_covariance(observations, rng=rng, train=False)
        rng, sample_rng = jax.random.split(rng)
        noise = dist.sample(seed=sample_rng)
        
        # Apply noise scaling if configured
        dgn_annealing_timescale = self.config.get("dgn_annealing_timescale", None)
        if total_env_steps is not None and dgn_annealing_timescale is not None:
            scale = jnp.exp(-max(1, total_env_steps) / dgn_annealing_timescale)
            noise = noise * scale
        
        # Remove batch dimension if single observation
        if single_obs:
            noise = noise.squeeze(0)
        
        return noise

    @staticmethod
    @partial(jax.jit, static_argnames=("entropy_coef",))
    def _train_step(
        state: JaxRLTrainState, 
        batch: Batch, 
        rng: PRNGKey,
        entropy_coef: float = 0.0
    ) -> Tuple[JaxRLTrainState, dict]:
        """Single gradient update step on covariance network."""
        rng, dropout_rng = jax.random.split(rng)
        
        def loss_fn(params):
            # Forward pass through covariance network
            dist = state.apply_fn(
                params,
                batch["observations"],
                train=True,
                rngs={"dropout": dropout_rng},
            )
            
            # Negative log-likelihood of action deltas under learned distribution
            action_deltas = batch["action_deltas"]
            nll = -dist.log_prob(action_deltas).mean()
            entropy = dist.entropy().mean()
            
            # Loss with optional entropy regularization
            if entropy_coef > 0:
                loss = nll - entropy_coef * entropy
            else:
                loss = nll
            
            # Compute mean covariance for logging
            covariance = dist.covariance()
            mean_covariance = jnp.mean(jnp.diagonal(covariance, axis1=-2, axis2=-1))
            
            info = {
                "covariance_loss": loss,
                "covariance_nll": nll,
                "covariance_entropy": entropy,
                "mean_covariance": mean_covariance,
                "mean_delta_norm": jnp.mean(jnp.linalg.norm(action_deltas, axis=-1)),
            }
            
            return loss, info
        
        grads, info = jax.grad(loss_fn, has_aux=True)(state.params)
        
        # Update state manually since we're using direct network
        updates, new_opt_state = state.txs.update(grads, state.opt_states, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_states=new_opt_state,
            rng=rng
        )
        
        return new_state, info
    
    def compute_deltas(self, agent, batch_size: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute deltas between expert and policy actions.
        
        Returns:
            observations: Sampled observations from demo dataset
            action_deltas: Expert actions - Policy actions
        """
        if batch_size is None:
            batch_size = len(self.demo_dataset["observations"])
        
        # Sample from demo dataset
        indices = np.random.choice(len(self.demo_dataset["observations"]), size=batch_size, replace=False)
        observations = self.demo_dataset["observations"][indices]
        expert_actions = self.demo_dataset["actions"][indices]
        
        # Get deterministic policy actions
        predicted_actions = agent.forward_policy(observations, train=False).mode()
        
        # Compute deltas: expert - policy
        action_deltas = expert_actions - predicted_actions
        
        return observations, action_deltas
    
    def update(self, agent, step: int) -> Tuple["DGNModule", dict]:
        """Update covariance network on demonstration data."""
        # Compute deltas for entire dataset
        states, action_deltas = self.compute_deltas(agent)
        n_samples = states.shape[0]
        
        state = self.state
        all_infos = {}
        
        dgn_entropy_coef = self.config["dgn_entropy_coef"]
        
        for epoch in range(self.config["dgn_cov_train_epochs"]):
            epoch_infos = []
            
            # Shuffle indices
            rng, shuffle_rng = jax.random.split(state.rng)
            shuffled_indices = jax.random.permutation(shuffle_rng, n_samples)
            
            # Train on batches
            for batch_idx in range(0, n_samples, self.config["dgn_batch_size"]):
                end_idx = min(batch_idx + self.config["dgn_batch_size"], n_samples)
                batch_indices = shuffled_indices[batch_idx:end_idx]
                
                batch = {
                    "observations": states[batch_indices],
                    "action_deltas": action_deltas[batch_indices],  # Store deltas in "actions" field
                }
                
                # Update on batch
                rng, step_rng = jax.random.split(rng)
                state, info = self._train_step(
                    state, 
                    batch, 
                    step_rng,
                    entropy_coef=dgn_entropy_coef
                )
                epoch_infos.append(info)
            
            # Average info across batches in last epoch
            if epoch == self.config["dgn_cov_train_epochs"] - 1:
                all_infos = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_infos)
        
        # Add step to logged info
        all_infos["dgn_update_step"] = step
        
        # Add noise scale if annealing is configured
        dgn_annealing_timescale = self.config.get("dgn_annealing_timescale", None)
        if dgn_annealing_timescale is not None:
            noise_scale = float(jnp.exp(-max(1, step) / dgn_annealing_timescale))
            all_infos["noise_scale"] = noise_scale
        
        # Return updated module
        return self.replace(state=state, step=self.step + 1), all_infos
    
    def dgn_update(self, agent, env_step: int) -> Tuple["DGNModule", dict]:
        """Update DGN if it's time based on update interval."""
        if env_step % self.config["dgn_update_interval"] == 0:
            return self.update(agent, env_step)
        return self, {}
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        demo_dataset: dict,
        # Network architecture
        covariance_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "dropout_rate": 0.5,
        },
        # Optimizer
        dgn_covariance_optimizer_kwargs: dict = {
            "learning_rate": 1e-4,
            "weight_decay": 3e-2,
        },
        **kwargs,
    ):
        """Create a new DGN module."""
        # Create config from kwargs - all values should be provided from dgn config
        config = ConfigDict(kwargs)
        config.action_dim = actions.shape[-1]
        config.observation_dim = observations.shape[-1] if observations.ndim > 1 else observations.shape[0]
        config.demo_dataset_size = len(demo_dataset["observations"])
        
        # Store architecture configs for reference
        config.covariance_network_kwargs = covariance_network_kwargs
        config.dgn_covariance_optimizer_kwargs = dgn_covariance_optimizer_kwargs
        
        # Store flattened versions for backward compatibility
        config.hidden_dims = covariance_network_kwargs.get("hidden_dims", [256, 256])
        config.dropout_rate = covariance_network_kwargs.get("dropout_rate", 0.5)
        
        # Define network  
        covariance_def = CovarianceNetwork(
            hidden_dims=tuple(covariance_network_kwargs["hidden_dims"]),
            output_dim=config.action_dim,
            dropout_rate=covariance_network_kwargs["dropout_rate"],
            cov_diagonal_eps=config["cov_diagonal_eps"],
        )
        
        # Initialize params directly with the network (no ModuleDict)
        rng, init_rng, dropout_rng = jax.random.split(rng, 3)
        init_rng_dict = {"params": init_rng, "dropout": dropout_rng}
        params = covariance_def.init(init_rng_dict, observations)
        
        # Create optimizer
        tx = make_optimizer(**dgn_covariance_optimizer_kwargs)
        
        # Create state
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=covariance_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )
        
        config = flax.core.FrozenDict(config)
        
        return cls(state=state, config=config, demo_dataset=demo_dataset) 