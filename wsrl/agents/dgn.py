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
        # Use MLP for hidden layers
        x = MLP(
            hidden_dims=self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate if train else 0.0,
        )(observations, train=train)
        
        # Output layer - outputs flat vector of params
        # For d-dimensional action space, we need d*(d+1)/2 elements
        num_params = self.output_dim * (self.output_dim + 1) // 2
        cholesky_elements = nn.Dense(num_params)(x)
        
        return cholesky_elements


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
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Convert raw network output to lower triangular Cholesky matrix."""
        cholesky_elements = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            train=train,
            name="covariance",
            rngs={"dropout": rng} if train else {},
        )
        
        action_dim = self.config["action_dim"]
        
        def params_to_tril(flat_params):
            # Handle potential numerical issues
            flat_params = jnp.nan_to_num(flat_params, nan=0.0, posinf=10.0, neginf=-10.0)
            
            L = jnp.zeros((action_dim, action_dim))
            tril_indices = jnp.tril_indices(action_dim)
            L = L.at[tril_indices].set(flat_params)
            
            # Apply softplus to diagonal to ensure positive
            diag_indices = jnp.arange(action_dim)
            L = L.at[diag_indices, diag_indices].set(
                nn.softplus(L[diag_indices, diag_indices]) + self.config.get("cov_diagonal_eps", 1e-5)
            )
            
            return L
        
        # Apply to batch using vmap
        L = jax.vmap(params_to_tril)(cholesky_elements)
        return L

    def covariance_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Train covariance network to minimize negative log-likelihood on demonstration data.
        Following the official implementation, we fit N(0, Σ) to the deltas between
        expert actions and current policy actions.
        """
        demo_states = batch["observations"]
        demo_actions = batch["actions"]
        
        rng, policy_rng = jax.random.split(rng)
        policy_dist = self.forward_policy(demo_states, rng=policy_rng, train=False)
        predicted_actions = policy_dist.mode()
        
        action_deltas = demo_actions - predicted_actions
        
        # Get covariance from DGN network
        rng, cov_rng = jax.random.split(rng)
        L = self.forward_covariance(
            demo_states, 
            rng=cov_rng,
            grad_params=params, 
            train=True
        )
        
        # zero-mean distribution
        batch_size = demo_states.shape[0]
        action_dim = self.config["action_dim"]
        zero_mean = jnp.zeros((batch_size, action_dim))
        delta_dist = distrax.MultivariateNormalTri(
            loc=zero_mean,
            scale_tri=L
        )
        
        # Compute negative log-likelihood
        log_probs = delta_dist.log_prob(action_deltas)
        nll = -jnp.mean(log_probs)
        
        # Compute entropy for regularization
        entropy = delta_dist.entropy().mean()
        
        # Apply entropy regularization if configured
        entropy_coef = self.config.get("dgn_entropy_coef", 0.0)
        loss = nll
        if entropy_coef > 0:
            loss = loss - entropy_coef * entropy
        
        # Get covariance for logging
        covariance = delta_dist.covariance()
        mean_covariance = jnp.mean(jnp.diagonal(covariance, axis1=-2, axis2=-1))
        
        info = {
            "covariance_loss": loss,
            "covariance_nll": nll,
            "covariance_entropy": entropy,
            "mean_covariance": mean_covariance,
            "mean_delta_norm": jnp.mean(jnp.linalg.norm(action_deltas, axis=-1)),
        }
        
        return loss, info
    
    @overrides
    def loss_fns(self, batch):
        """Include all losses including covariance (will be filtered by networks_to_update)."""
        losses = super().loss_fns(batch)
        losses["covariance"] = partial(self.covariance_loss_fn, batch)
        return losses


    def update_covariance(self, pmap_axis: Optional[str] = None):
        """Update covariance network on entire demonstration data for multiple epochs."""
        assert self.demo_dataset is not None, "No demonstration data provided. Initialize agent with demo_dataset."
            
        batch_size = self.config.get("dgn_batch_size", 128)
            
        dataset_size = len(self.demo_dataset["observations"])
        indices = jnp.arange(dataset_size)
            
        n_epochs = self.config.get("dgn_cov_train_epochs", 1)
            
        agent = self
        all_infos = {}
            
        for epoch in range(n_epochs):
            # Shuffle indices for each epoch
            rng, shuffle_rng = jax.random.split(agent.state.rng)
            shuffled_indices = jax.random.permutation(shuffle_rng, indices)
            
            # Create batches
            num_batches = dataset_size // batch_size
            epoch_infos = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = shuffled_indices[start_idx:end_idx]
                
                # Create batch from demo_dataset
                demo_batch = {
                    "observations": self.demo_dataset["observations"][batch_indices],
                    "actions": self.demo_dataset["actions"][batch_indices],
                }
                
                # Update on this batch
                agent, info = agent.update(
                    demo_batch,
                    pmap_axis=pmap_axis,
                    networks_to_update=frozenset({"covariance"}),
                )
                epoch_infos.append(info)
            
            # Average info across batches in epoch
            if epoch == n_epochs - 1:
                # Keep last epoch's averaged info
                all_infos = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_infos)
                
        return agent, all_infos
    
    def get_noise_scale(self, total_env_steps: int) -> float:
        """
        Compute noise scale based on annealing or success-based shutoff.
        """
        if self.config.get("dgn_annealing_timescale") is not None:
            # Apply exponential annealing
            scale = jnp.exp(-total_env_steps / self.config["dgn_annealing_timescale"])
            return scale
        
        return 1.0
    
    def should_shutoff_noise(self) -> bool:
        """
        Check if noise should be shut off based on success rate.
        This is called outside of JIT-compiled functions.
        """
        if self.config.get("dgn_shutoff_success_threshold") is not None:
            if len(self.success_history) >= self.config["dgn_shutoff_epochs"]:
                recent_success_rate = np.mean(
                    self.success_history[-self.config["dgn_shutoff_epochs"]:]
                )
                if recent_success_rate >= self.config["dgn_shutoff_success_threshold"]:
                    return True
        return False
    
    @overrides
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
        
        Following the official implementation:
        - action = policy_mean + perturbation
        - perturbation ~ N(0, Σ)
        
        Args:
            observations: Input observations
            seed: Random seed for sampling
            argmax: If True, use deterministic policy (for evaluation)
            total_env_steps: Total environment steps (required for exploration with annealing)
        """
        if argmax:
            # For evaluation, use deterministic policy
            actions = super().sample_actions(observations, seed=seed, argmax=True)
            # Convert JAX array to NumPy array for environment compatibility
            actions = np.asarray(actions)
            return actions
        
        # Check if noise should be shut off (outside JIT)
        if self.should_shutoff_noise():
            # Use deterministic policy when shutoff is active
            actions = super().sample_actions(observations, seed=seed, argmax=True)
            # Convert JAX array to NumPy array for environment compatibility
            actions = np.asarray(actions)
            return actions
        
        # Get policy mean and perturbation distribution
        policy_mean, perturbation_dist = self.forward_sampling_policy(
            observations, 
            rng=seed,
            apply_noise_scaling=True,  # Apply annealing/shutoff
            total_env_steps=total_env_steps,
            train=False
        )
        
        # Sample perturbation
        if seed is not None:
            rng, sample_rng = jax.random.split(seed)
        else:
            sample_rng = self.state.rng
            
        perturbation = perturbation_dist.sample(seed=sample_rng)
        
        # Add perturbation to policy mean
        actions = policy_mean + perturbation
        
        # Clip actions to valid range
        actions = jnp.clip(actions, -1.0, 1.0)
        
        # Convert JAX array to NumPy array for environment compatibility
        actions = np.asarray(actions)
        
        # Ensure actions is a flat array (not nested) for single environment
        if actions.ndim > 1:
            actions = actions.squeeze()
        
        return actions
    
    def update_success_history(self, episode_success: bool) -> "DGNAgent":
        """Update success history for shutoff mechanism."""
        new_history = self.success_history + [float(episode_success)]
        # Keep only recent history
        max_history = self.config.get("dgn_shutoff_epochs", 10) * 2
        if len(new_history) > max_history:
            new_history = new_history[-max_history:]
        return self.replace(success_history=new_history)
    
    def forward_sampling_policy(
        self,
        observations: Data,
        *,
        rng: Optional[PRNGKey] = None,
        grad_params: Optional[Params] = None,
        apply_noise_scaling: bool = True,
        total_env_steps: Optional[int] = None,
        train: bool = False,
        covariance_train: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray, distrax.Distribution]:
        """
        Get policy mean and perturbation distribution for DGN exploration.
        
        Following the official implementation, we return:
        - policy_mean: The deterministic action from the policy
        - perturbation_dist: N(0, Σ) distribution for exploration noise
        
        The final action is: policy_mean + sample from perturbation_dist
        """
        covariance_rng = None
        if train or covariance_train:
            assert rng is not None, "Must specify rng when training"
            rng, covariance_rng = jax.random.split(rng)
            covariance_train = train
            
        # Get policy mean from SAC
        policy_mean = self.forward_policy(
            observations, 
            rng=rng, 
            grad_params=grad_params, 
            train=train
        ).mode()
        
        # Get covariance from DGN network
        L = self.forward_covariance(
            observations, 
            rng=covariance_rng, 
            grad_params=grad_params, 
            train=covariance_train
        )
        
        # Apply noise scaling if requested
        if apply_noise_scaling:
            if total_env_steps is None:
                raise ValueError("total_env_steps required when apply_noise_scaling=True")
            noise_scale = self.get_noise_scale(total_env_steps)
            L = L * noise_scale
        
        # Create zero-mean perturbation distribution (following official implementation)
        batch_size = observations.shape[0]
        action_dim = self.config["action_dim"]
        zero_mean = jnp.zeros((batch_size, action_dim))
        perturbation_dist = distrax.MultivariateNormalTri(
            loc=zero_mean,
            scale_tri=L
        )
        
        return policy_mean, perturbation_dist
    
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
        config.action_dim = actions.shape[-1]

        
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
            output_dim=config.action_dim,
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
        
        # Create a separate RNG key for dropout
        rng, dropout_rng = jax.random.split(rng)
        init_rng_dict = {
            "params": init_rng,
            "dropout": dropout_rng
        }
        
        params = model_def.init(
            init_rng_dict,
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
        
        # Add missing SAC parameters if not provided
        if "max_target_backup" not in config:
            config.max_target_backup = False
        if "n_actions" not in config:
            config.n_actions = 10
        
        config = flax.core.FrozenDict(config)
        
        return cls(
            state=state,
            config=config,
            demo_dataset=demo_dataset,
            success_history=[],
        ) 




    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
    ) -> Tuple["DGNAgent", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.
        
        For DGN, the covariance network is updated separately via update_covariance()
        on demonstration data, not on RL replay buffer data.
        """
        # Override parent to handle covariance updates properly
        return super().update(batch, pmap_axis=pmap_axis, networks_to_update=networks_to_update) 