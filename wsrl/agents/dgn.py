"""
Implementation of Data-Guided Noise (DGN) for exploration in RL.
Based on https://arxiv.org/pdf/2506.07505
"""
import copy
from functools import partial
from typing import Optional, Tuple

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from overrides import overrides

from wsrl.agents.sac import SACAgent
from wsrl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.networks.actor_critic_nets import Critic, Policy, ensemblize
from wsrl.networks.lagrange import GeqLagrangeMultiplier
from wsrl.networks.mlp import MLP


class CovarianceNetwork(nn.Module):
    """
    Network that outputs Cholesky decomposition of covariance matrix.
    """
    hidden_dims: Tuple[int, ...] = (256, 256)
    output_dim: int = None  # action_dim, will be set in setup
    dropout_rate: float = 0.5
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = observations
        
        # MLP layers with dropout
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            if train:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        # Output layer - outputs lower triangular matrix elements
        # For d-dimensional action space, we need d*(d+1)/2 elements
        num_elements = self.output_dim * (self.output_dim + 1) // 2
        cholesky_elements = nn.Dense(num_elements)(x)
        
        # Construct lower triangular matrix
        batch_size = observations.shape[0]
        L = jnp.zeros((batch_size, self.output_dim, self.output_dim))
        
        # Fill lower triangular matrix
        idx = 0
        for i in range(self.output_dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal elements must be positive - use softplus
                    L = L.at[:, i, j].set(nn.softplus(cholesky_elements[:, idx]) + 1e-6)
                else:
                    L = L.at[:, i, j].set(cholesky_elements[:, idx])
                idx += 1
        
        return L


class DGNAgent(SACAgent):
    """
    Data-Guided Noise agent that uses demonstration data to guide exploration.
    
    Based on "Reinforcement Learning via Implicit Imitation Guidance" (https://arxiv.org/pdf/2506.07505)
    
    DGN learns a state-dependent covariance matrix from demonstration data and uses it to
    guide exploration during online RL. The key insight is that demonstrations are most
    useful for identifying which actions should be explored, rather than forcing the
    policy to take certain actions.
    
    Usage Example:
    ```python
    # Create DGN agent with demonstration dataset
    agent = DGNAgent.create(
        rng=rng,
        observations=observations,
        actions=actions,
        encoder_def=encoder,
        demo_dataset={"observations": demo_obs, "actions": demo_actions},
        dgn_update_interval=1000,
        dgn_annealing_timescale=30000,  # or use success-based shutoff
        dgn_shutoff_success_threshold=0.5,
        dgn_shutoff_epochs=10,
    )
    
    # Training loop
    total_env_steps = 0
    for step in range(num_steps):
        # Sample action with data-guided noise
        action = agent.sample_actions(
            observation, 
            seed=rng, 
            total_env_steps=total_env_steps
        )
        
        # Take environment step
        next_obs, reward, done, info = env.step(action)
        total_env_steps += 1
        
        # Store transition in replay buffer
        replay_buffer.add(observation, action, reward, next_obs, done)
        
        # Update agent (standard SAC update)
        batch = replay_buffer.sample(batch_size)
        agent, update_info = agent.update(batch)
        
        # Update covariance network every N steps
        if step % agent.config["dgn_update_interval"] == 0:
            demo_batch = sample_demo_batch(agent.demo_dataset, batch_size)
            agent, cov_info = agent.update_covariance(demo_batch)
        
        # Track success for shutoff mechanism (returns new agent)
        if done:
            agent = agent.update_success_history(info.get("success", False))
    ```
    """
    
    demo_dataset: Optional[dict] = nonpytree_field(default=None)
    success_history: list = nonpytree_field(default_factory=list)
    
    def forward_covariance(
        self,
        observations: Data,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass for covariance network that outputs Cholesky decomposition.
        """
        L = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            train=train,
            name="covariance",
        )
        return L
    
    def covariance_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Train covariance network to minimize negative log-likelihood on demonstration data.
        """
        # Get demonstration data
        demo_states = batch["observations"]
        demo_actions = batch["actions"]
        
        # Get the sampling policy distribution
        rng, policy_rng = jax.random.split(rng)
        sampling_dist = self.forward_sampling_policy(
            demo_states,
            rng=policy_rng,
            grad_params=params,  # Pass grad_params for covariance network gradients
            apply_noise_scaling=False,  # No scaling during training
            train=False,  # Policy in eval mode (no gradients)
            covariance_train=True  # Covariance in train mode (with gradients)
        )
        
        # Compute negative log-likelihood
        log_probs = sampling_dist.log_prob(demo_actions)
        loss = -jnp.mean(log_probs)
        
        # Get covariance for logging
        covariance = sampling_dist.covariance()
        mean_covariance = jnp.mean(jnp.diagonal(covariance, axis1=-2, axis2=-1))
        
        info = {
            "covariance_loss": loss,
            "mean_covariance": mean_covariance,
            "covariance_log_det": jnp.mean(sampling_dist.log_det_covariance()),
        }
        
        return loss, info
    
    @overrides
    def loss_fns(self, batch):
        """Add covariance loss to the standard SAC losses."""
        losses = super().loss_fns(batch)
        # Note: covariance loss is computed separately on demo data
        return losses
    
    def update_covariance(self, demo_batch: Optional[Batch] = None, pmap_axis: Optional[str] = None):
        """Update covariance network on demonstration data."""
        if demo_batch is None:
            if self.demo_dataset is None:
                raise ValueError("No demonstration data provided. Either pass demo_batch or initialize agent with demo_dataset.")
            # TODO: Sample from self.demo_dataset
            raise NotImplementedError("Automatic sampling from demo_dataset not yet implemented. Please provide demo_batch.")
        
        rng, key = jax.random.split(self.state.rng)
        
        # Create loss function
        loss_fn = partial(self.covariance_loss_fn, demo_batch)
        
        # Update only covariance network
        loss_fns = {
            "covariance": loss_fn,
            # Dummy losses for other networks
            "actor": lambda params, rng: (0.0, {}),
            "critic": lambda params, rng: (0.0, {}),
            "temperature": lambda params, rng: (0.0, {}),
        }
        
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        
        # Update RNG
        new_state = new_state.replace(rng=rng)
        
        return self.replace(state=new_state), info["covariance"]
    
    def get_noise_scale(self, total_env_steps: int) -> float:
        """
        Compute noise scale based on annealing or success-based shutoff.
        """
        if self.config.get("dgn_shutoff_success_threshold") is not None:
            # Check if we should shut off based on success rate
            if len(self.success_history) >= self.config["dgn_shutoff_epochs"]:
                recent_success_rate = np.mean(
                    self.success_history[-self.config["dgn_shutoff_epochs"]:]
                )
                if recent_success_rate >= self.config["dgn_shutoff_success_threshold"]:
                    return 0.0
        
        if self.config.get("dgn_annealing_timescale") is not None:
            # Apply exponential annealing
            scale = jnp.exp(-total_env_steps / self.config["dgn_annealing_timescale"])
            return scale
        
        return 1.0
    
    @overrides
    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        total_env_steps: Optional[int] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions using data-guided noise for exploration.
        
        Args:
            observations: Input observations
            seed: Random seed for sampling
            argmax: If True, use deterministic policy (for evaluation)
            total_env_steps: Total environment steps (required for exploration with annealing)
        """
        if argmax:
            # For evaluation, use deterministic policy
            return super().sample_actions(observations, seed=seed, argmax=True)
        
        # Get the DGN sampling distribution
        sampling_dist = self.forward_sampling_policy(
            observations, 
            rng=seed,
            apply_noise_scaling=True,  # Apply annealing/shutoff
            total_env_steps=total_env_steps,
            train=False
        )
        
        # Sample actions
        if seed is not None:
            rng, sample_rng = jax.random.split(seed)
        else:
            sample_rng = self.state.rng
            
        actions = sampling_dist.sample(seed=sample_rng)
        
        # Clip actions to valid range
        actions = jnp.clip(actions, -1.0, 1.0)
        
        return actions
    
    def update_success_history(self, episode_success: bool) -> "DGNAgent":
        """Update success history for shutoff mechanism."""
        new_history = self.success_history + [float(episode_success)]
        # Keep only recent history
        max_history = self.config.get("dgn_shutoff_epochs", 10) * 2
        if len(new_history) > max_history:
            new_history = new_history[-max_history:]
        return self.replace(success_history=new_history)
    
    @partial(jax.jit, static_argnames=("apply_noise_scaling", "train", "covariance_train"))
    def forward_sampling_policy(
        self,
        observations: Data,
        *,
        rng: Optional[PRNGKey] = None,
        grad_params: Optional[Params] = None,
        apply_noise_scaling: bool = True,
        total_env_steps: Optional[int] = None,
        train: bool = False,
        covariance_train: Optional[bool] = None,
    ) -> distrax.Distribution:
        """
        Forward pass for the DGN sampling policy π_sampling(a|s) = N(μ_θ(s), Σ_φ(s)).
        
        This combines the SAC policy mean with the learned DGN covariance.
        
        Args:
            observations: Input observations
            rng: Random key for policy forward pass
            grad_params: Optional parameters for gradient computation
            apply_noise_scaling: Whether to apply annealing/shutoff scaling
            total_env_steps: Total environment steps for annealing (required if apply_noise_scaling=True)
            train: Whether in training mode for policy network
            covariance_train: Whether in training mode for covariance network (defaults to train)
            
        Returns:
            MultivariateNormalTri distribution for sampling
        """
        if covariance_train is None:
            covariance_train = train
            
        # Get policy mean from SAC
        policy_dist = self.forward_policy(observations, rng=rng, grad_params=grad_params, train=train)
        policy_mean = policy_dist.mode()
        
        # Get covariance from DGN network
        L = self.forward_covariance(observations, grad_params=grad_params, train=covariance_train)
        
        # Apply noise scaling if requested
        if apply_noise_scaling:
            if total_env_steps is None:
                raise ValueError("total_env_steps required when apply_noise_scaling=True")
            noise_scale = self.get_noise_scale(total_env_steps)
            L = L * noise_scale
        
        # Create multivariate normal sampling distribution
        sampling_dist = distrax.MultivariateNormalTri(
            loc=policy_mean,
            scale_tri=L
        )
        
        return sampling_dist
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        # DGN specific
        demo_dataset: Optional[dict] = None,
        covariance_network_kwargs: dict = {
            "hidden_dims": [128, 128],
            "dropout_rate": 0.5,
        },
        **kwargs,
    ):
        """
        Create a new DGN agent.
        """
        # Create config from all arguments
        config = ConfigDict(kwargs)
        
        # Set up encoders
        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "critic": copy.deepcopy(encoder_def),
            }
        
        # Define networks
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )
        
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic,
            encoder=encoders["critic"],
            network=critic_backbone,
        )(name="critic")
        
        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )
        
        # DGN covariance network
        covariance_def = CovarianceNetwork(
            output_dim=actions.shape[-1],
            **covariance_network_kwargs,
            name="covariance",
        )
        
        # Model Def
        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
            "covariance": covariance_def,
        }
        model_def = ModuleDict(networks)
        
        # Define optimizers
        txs = {
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
            "covariance": make_optimizer(**config.dgn_covariance_optimizer_kwargs),
        }
        
        # Initialize params
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
            covariance=[observations],
        )["params"]
        
        # Create state
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )
        
        if config.get("target_entropy", 0.0) >= 0.0:
            config.target_entropy = -actions.shape[-1]
        config = flax.core.FrozenDict(config)
        
        return cls(
            state=state,
            config=config,
            demo_dataset=demo_dataset,
            success_history=[],
        ) 