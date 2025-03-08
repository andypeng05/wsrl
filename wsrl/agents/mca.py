import chex
import jax
import jax.numpy as jnp
from wsrl.agents.sac import SACAgent
from wsrl.common.typing import Params, PRNGKey

class MCAgent(SACAgent):
    """Same agent as SAC, just using a different critic loss."""

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        # next_actions, next_actions_log_probs = self._compute_next_actions(
        #     batch, next_action_sample_key
        # )
        # # (batch_size, ) for sac, (batch_size, cql_n_actions) for cql

        # # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        # target_next_qs = self.forward_target_critic(
        #     batch["next_observations"],
        #     next_actions,
        #     rng=rng,
        # )  # (critic_ensemble_size, batch_size)

        # # Subsample if requested
        # if self.config["critic_subsample_size"] is not None:
        #     rng, subsample_key = jax.random.split(rng)
        #     subsample_idcs = jax.random.randint(
        #         subsample_key,
        #         (self.config["critic_subsample_size"],),
        #         0,
        #         self.config["critic_ensemble_size"],
        #     )
        #     target_next_qs = target_next_qs[subsample_idcs]

        # # Minimum Q across (subsampled) ensemble members
        # target_next_min_q = target_next_qs.min(axis=0)
        # chex.assert_equal_shape([target_next_min_q, next_actions_log_probs])
        # # (batch_size,) for sac, (batch_size, cql_n_actions) for cql

        # target_next_min_q = self._process_target_next_qs(
        #     target_next_min_q,
        #     next_actions_log_probs,
        # )

        # target_q = (
        #     batch["rewards"]
        #     + self.config["discount"] * batch["masks"] * target_next_min_q
        # )
        # chex.assert_shape(target_q, (batch_size,))

        predicted_qs = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=rng,
            grad_params=params,
        )
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )

        # MSE loss
        assert "mc_returns" in batch.keys()
        target_qs = batch["mc_returns"][None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "mc_returns": jnp.mean(batch["mc_returns"]),
        }

        return critic_loss, info
    

