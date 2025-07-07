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
        
        # MLP layers with optional dropout
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            
            # Only apply dropout during training
            if train and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        
        # Output layer - outputs flat vector of params
        # For d-dimensional action space, we need d*(d+1)/2 elements
        num_params = self.output_dim * (self.output_dim + 1) // 2
        cholesky_elements = nn.Dense(num_params)(x)
        
        return cholesky_elements


def _make_sampling_dist(policy_mean, L):
    return distrax.MultivariateNormalTri(loc=policy_mean, scale_tri=L)


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
            if flat_params.ndim == 2:
                # Batch of parameters: (batch_size, num_params)
                def single_params_to_tril(single_flat_params):
                    L = jnp.zeros((action_dim, action_dim))
                    tril_indices = jnp.tril_indices(action_dim)
                    L = L.at[tril_indices].set(single_flat_params)
                    diag_mask = jnp.eye(action_dim, dtype=bool)
                    L = jnp.where(diag_mask, nn.softplus(L) + 1e-6, L * 0.1)
                    return L
                return jax.vmap(single_params_to_tril)(flat_params)
            else:
                # Single set of parameters: (num_params,)
                L = jnp.zeros((action_dim, action_dim))
                tril_indices = jnp.tril_indices(action_dim)
                L = L.at[tril_indices].set(flat_params)
                diag_mask = jnp.eye(action_dim, dtype=bool)
                L = jnp.where(diag_mask, nn.softplus(L) + 1e-6, L * 0.1)
                return L
            
        L = params_to_tril(cholesky_elements)
        return L

    def covariance_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Train covariance network to minimize negative log-likelihood on demonstration data.
        Returns zero loss when not training covariance (e.g., during standard SAC updates).
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
            "covariance_log_det": jnp.mean(jnp.log(jnp.linalg.det(covariance))),
        }
        
        return loss, info
    
    @overrides
    def loss_fns(self, batch):
        """Include all losses including covariance (will be filtered by networks_to_update)."""
        losses = super().loss_fns(batch)
        losses["covariance"] = partial(self.covariance_loss_fn, batch)
        return losses


    def update_covariance(self, demo_batch, pmap_axis=None):
        """Update only the covariance network on demonstration data."""
        # Use the standard update mechanism, but only update covariance network
        return self.update(
            demo_batch, 
            pmap_axis=pmap_axis, 
            networks_to_update=frozenset({"covariance"}),
            update_covariance_flag=True,
        )
    
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
        
        # Convert JAX array to NumPy array for environment compatibility
        actions = np.asarray(actions)
        
        # Ensure actions is a flat array (not nested) TODO: check if this is needed
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
    ) -> distrax.Distribution:
        """
        Forward pass for the DGN sampling policy π_sampling(a|s) = N(μ_θ(s), Σ_φ(s)).
        This combines the SAC policy mean with the learned DGN covariance.
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
        # Create multivariate normal sampling distribution using pure function
        sampling_dist = _make_sampling_dist(policy_mean, L)
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




    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update", "update_covariance_flag"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
        update_covariance_flag: bool = False,
    ) -> Tuple["DGNAgent", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.
        
        Args:
            batch: Training batch
            pmap_axis: Axis for pmap
            networks_to_update: Networks to update for this call
            update_covariance_flag: If True, also update covariance network
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, key = jax.random.split(self.state.rng)

        # Determine which networks to actually update
        final_networks_to_update = networks_to_update
        if update_covariance_flag:
            final_networks_to_update = networks_to_update | {"covariance"}

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified networks
        assert final_networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {final_networks_to_update}"
        for key in loss_fns.keys() - final_networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in final_networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

        
    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis", "update_covariance_flag"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
        update_covariance_flag: bool = False,
    ) -> Tuple["DGNAgent", dict]:
        """
        Fast JITted high-UTD version of `.update` with DGN covariance support.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple["DGNAgent"], data: Tuple[Batch]):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(
                minibatch,
                pmap_axis=pmap_axis,
                networks_to_update=frozenset({"critic"}),
                update_covariance_flag=False,  # Never update covariance during critic updates
            )
            return (agent,), info

        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (utd_ratio, minibatch_size) + data.shape[1:])

        minibatches = jax.tree_map(make_minibatch, batch)

        (agent,), critic_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

        critic_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)
        del critic_infos["actor"]
        del critic_infos["temperature"]

        # Take one gradient descent step on the actor and temperature
        # Also handle covariance updates here
        networks_to_update = frozenset({"actor", "temperature"})

        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=networks_to_update,
            update_covariance_flag=update_covariance_flag,
        )
        del actor_temp_infos["critic"]

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos 

    def update_high_utd_with_step(
        self,
        batch: Batch,
        step: int,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DGNAgent", dict]:
        """
        Optimized high-UTD update with automatic covariance scheduling.
        For UTD > 1 training only.
        """
        # Determine if we should update covariance based on step
        dgn_update_interval = self.config.get("dgn_update_interval")
        update_covariance_flag = (
            dgn_update_interval is not None and 
            step % dgn_update_interval == 0
        )
        
        return self.update_high_utd(
            batch,
            utd_ratio=utd_ratio,
            pmap_axis=pmap_axis,
            update_covariance_flag=update_covariance_flag,
        ) 