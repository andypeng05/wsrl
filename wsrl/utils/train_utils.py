from collections.abc import Mapping
from typing import TypeVar

import numpy as np
import optax
from absl import flags
from flax.training import checkpoints
from jax import tree_util

FLAGS = flags.FLAGS
# only one of the following should be set to true
flags.DEFINE_bool(
    "not_load_value_last_layer",
    False,
    "don't load the last layer of the value function",
)
flags.DEFINE_bool(
    "not_load_value_last_bias", False, "don't load the last bias of the value function"
)


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


def restore_optimizer_state(
    opt_state: TX, restored: Mapping, del_alpha: bool = False
) -> TX:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    if del_alpha:
        del restored["actor"]["inner_state"]["0"]["0"]["nu"][
            "modules_cql_alpha_lagrange"
        ]
        del restored["actor"]["inner_state"]["0"]["0"]["mu"][
            "modules_cql_alpha_lagrange"
        ]
        del restored["temperature"]["inner_state"]["0"]["0"]["nu"][
            "modules_cql_alpha_lagrange"
        ]
        del restored["temperature"]["inner_state"]["0"]["0"]["mu"][
            "modules_cql_alpha_lagrange"
        ]
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
        del_alpha = False
        if "antmaze" in checkpoint_path:
            if "calql" in checkpoint_path or "cql" in checkpoint_path:
                if FLAGS.agent not in ("calql", "cql"):
                    del_alpha = True
        opt_states = restore_optimizer_state(
            agent.state.opt_states, opt_states, del_alpha
        )
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


def sac_value_loader(agent, checkpoint_path):
    # unfreeze the params from target agent
    params = agent.state.params
    target_params = agent.state.target_params

    # restore from checkpoint
    restored_agent = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    all_keys = restored_agent["state"]["params"]["modules_critic"].keys()
    if FLAGS.not_load_value_last_layer:
        # not load the 'Dense_0' key
        keys_to_load = ("network",)
    elif FLAGS.not_load_value_last_bias:
        keys_to_load = ("network", "Dense_0/kernel")
    else:
        keys_to_load = tuple(all_keys)
        assert "network" in all_keys
        assert "Dense_0" in all_keys
    for k in keys_to_load:
        if "/" in k:
            # nested keys
            high_k, low_k = k.split("/")
            params["modules_critic"][high_k][low_k] = restored_agent["state"]["params"][
                "modules_critic"
            ][high_k][low_k]
            target_params["modules_critic"][high_k][low_k] = restored_agent["state"][
                "target_params"
            ]["modules_critic"][high_k][low_k]
        else:
            params["modules_critic"][k] = restored_agent["state"]["params"][
                "modules_critic"
            ][k]
            target_params["modules_critic"][k] = restored_agent["state"][
                "target_params"
            ]["modules_critic"][k]

    # restore optimizer states
    opt_states = dict(agent.state.opt_states)  # hard-copy
    opt_states["critic"] = restored_agent["state"]["opt_states"]["critic"]
    opt_states = restore_optimizer_state(agent.state.opt_states, opt_states)

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


def iql_policy_loader(agent, checkpoint_path):
    # unfreeze the params from target agent
    params = agent.state.params
    target_params = agent.state.target_params

    # restore from checkpoint
    restored_agent = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    params["modules_actor"] = restored_agent["state"]["params"]["modules_actor"]
    target_params["modules_actor"] = restored_agent["state"]["target_params"][
        "modules_actor"
    ]

    # params = agent.state.params.copy(
    #     add_or_replace={
    #         "modules_actor":restored_agent["state"]["params"]["modules_actor"],
    #     }
    # )
    # target_params = agent.state.target_params.copy(
    #     add_or_replace={
    #         "modules_actor":restored_agent["state"]["target_params"]["modules_actor"],
    #     }
    # )

    # put the params back into the agent
    agent = agent.replace(
        state=agent.state.replace(
            params=params,
            target_params=target_params,
            step=restored_agent["state"]["step"],
        )
    )
    return agent


def iql_value_loader(agent, checkpoint_path):
    # unfreeze the params from target agent
    params = agent.state.params
    target_params = agent.state.target_params

    # restore from checkpoint
    restored_agent = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    params["modules_critic"] = restored_agent["state"]["params"]["modules_critic"]
    target_params["modules_critic"] = restored_agent["state"]["target_params"][
        "modules_critic"
    ]
    params["modules_value"] = restored_agent["state"]["params"]["modules_value"]
    target_params["modules_value"] = restored_agent["state"]["target_params"][
        "modules_value"
    ]

    # put the params back into the agent
    agent = agent.replace(
        state=agent.state.replace(
            params=params,
            target_params=target_params,
            step=restored_agent["state"]["step"],
        )
    )
    return agent


def step_loader(agent, checkpoint_path):
    # only restore the step
    restored_agent = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    agent = agent.replace(
        state=agent.state.replace(step=restored_agent["state"]["step"])
    )
    return agent


pretrained_loaders = dict(
    sac_policy_loader=sac_policy_loader,
    sac_value_loader=sac_value_loader,
    iql_policy_loader=iql_policy_loader,
    iql_value_loader=iql_value_loader,
    step_loader=step_loader,
)
