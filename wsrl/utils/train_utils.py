from collections.abc import Mapping

import numpy as np
from absl import flags


from jax import tree_util
from flax.training import checkpoints
from typing import TypeVar
import optax


FLAGS = flags.FLAGS


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            if "observation" in key and "visual" in FLAGS.env:
                concatenated[key] = np.concatenate(
                    [batch[key] for batch in batches], axis=0
                ).astype(np.uint8)
            else:
                concatenated[key] = np.concatenate(
                    [batch[key] for batch in batches], axis=0
                ).astype(np.float32)
    return concatenated


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        if isinstance(batch[key], Mapping):
            # to index into batch["observations"]["image"], etc.
            indexed[key] = index_batch(batch[key], indices)
        else:
            indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch["rewards"].shape[0], size=size)
    return index_batch(batch, indices)

TX = TypeVar("TX", bound=optax.OptState)

def restore_optimizer_state(opt_state: TX, restored: Mapping) -> TX:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    return tree_util.tree_unflatten(
        tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
    )

def sac_policy_loader(agent, checkpoint_path):
    # unfreeze the params from target agent
    params = agent.state.params
    target_params = agent.state.target_params

    # restore from checkpoint
    restored_agent = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    params["modules_actor"] = restored_agent["state"]["params"]["modules_actor"]
    target_params["modules_actor"] = restored_agent["state"]["target_params"][
        "modules_actor"
    ]
    try:
        params["modules_temperature"] = restored_agent["state"]["params"][
            "modules_temperature"
        ]
        target_params["modules_temperature"] = restored_agent["state"]["target_params"][
            "modules_temperature"
        ]
    except KeyError:
        # if temperature not in checkpoint, check we are loading the IQL checkpoint
        # and don't load the temperature
        assert "modules_value" in restored_agent["state"]["params"]
        print("Temperature not found in IQL checkpoint, not loading temperature")
        pass

    # restore optimizer states
    opt_states = dict(agent.state.opt_states)  # hard-copy
    opt_states["actor"] = restored_agent["state"]["opt_states"]["actor"]
    try:
        opt_states["temperature"] = restored_agent["state"]["opt_states"]["temperature"]
        opt_states = restore_optimizer_state(agent.state.opt_states, opt_states)
    except KeyError:
        # if temperature not in checkpoint, check we are loading the IQL checkpoint
        # and don't load the temperature
        assert "value" in restored_agent["state"]["opt_states"]
        print("Temperature not found in IQL checkpoint, not loading temperature")
        print("Also not loading optimizer state because we can't map over it")
        opt_states = agent.state.opt_states  # restore the fresh optimizer state
        pass

    # put the params back into the agent
    agent = agent.replace(
        state=agent.state.replace(
            params=params,
            target_params=target_params,
            step=restored_agent["state"]["step"],
            opt_states=opt_states,
        )
    )
    return agent