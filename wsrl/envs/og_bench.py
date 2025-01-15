import collections
from typing import Optional

import gym
import gymnasium
import numpy as np
import ogbench
from numba import jit

def is_og_bench_env(env_name: str) -> bool:
    if "navigate" in env_name:
        return True
    if "stitch" in env_name:
        return True
    if "explore" in env_name:
        return True
    if "cube" in env_name:
        return True
    if "scene" in env_name:
        return True
    if "puzzle" in env_name:
        return True
    return False


def choose_og_bench_task_id(env_name):
    if "maze" in env_name:
        task_id = 1
    elif "cube" in env_name:
        task_id = 2
    elif "scene" in env_name:
        task_id = 3
    elif "puzzle-3x3" in env_name:
        task_id = 4
    elif "puzzle" in env_name:
        task_id = 5
    else:
        raise ValueError(f"Invalid OG Bench env name: {env_name}")
    return task_id


def make_og_bench_env(**kwargs):
    """wrapper"""
    env, train_ds, val_ds = make_og_bench_env_and_datasets(**kwargs)
    del train_ds, val_ds
    return env


def make_og_bench_datasets(**kwargs):
    """wrapper"""
    env, train_ds, val_ds = make_og_bench_env_and_datasets(**kwargs)
    del train_ds["terminals"]
    del val_ds["terminals"]
    return train_ds, val_ds


def make_og_bench_datasets_with_mc(
    env_name,
    task_id: int = 1,
    reward_scale: Optional[float] = None,
    reward_bias: Optional[float] = None,
    gamma: float = 0.99,
    seed: int = 0,
):
    """add MC return to the datasets"""
    env, train_ds, val_ds = make_og_bench_env_and_datasets(
        env_name=env_name,
        task_id=task_id,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        seed=seed,
    )

    # add MC return
    train_ds_mc = ogbench_dataset_and_calc_mc(
        env,
        dataset=train_ds,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        gamma=gamma,
        infinite_horizon=False,
    )
    val_ds_mc = ogbench_dataset_and_calc_mc(
        env,
        dataset=val_ds,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        gamma=gamma,
        infinite_horizon=False,
    )

    for ds in (train_ds_mc, val_ds_mc):
        del ds["terminals"]

    return train_ds_mc, val_ds_mc


def ogbench_dataset_and_calc_mc(
    env,
    dataset,
    reward_scale,
    reward_bias,
    gamma,
    is_sparse_reward=True,
    infinite_horizon=False,
):
    """
    Add MC return calculations to the dataset.
    Based on

    But the difference with jaxrl_m/envs/d4rl.py:qlearning_dataset_and_calc_mc
    is that in OGBench environments, each "episode" (during data-collection)
    could be targetting different goals, so in the single-task setting they are
    really separate episodes.

    In single task setting, each collection-episode (with max_episode_steps) either
    never reaches the goal, or reaches the goal at some intermediate timestep,
    and then continues to reach other goals until max_episode_steps.

    E.g.
    dones: [0, 0, .., 0, 1, 1, 1, 1, 0, 0, ..., 0]
    Ep # : [0, 0, .., 0, 0, 0, 0, 0, 1, 1, ..., 1]

    There may be consecutive transitions where the done signal is True, so we
    consider the episode to last until the last done signal.
    """
    # lazy imports
    from wsrl.envs.env_common import calc_return_to_go
    from wsrl.utils.train_utils import concatenate_batches

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # separate the dataset into episodes
    # first, just break up the dataset by max_episode_steps
    assert N % env.spec.max_episode_steps == 0
    n_episodes = N // env.spec.max_episode_steps
    for i in range(n_episodes):
        start_idx = i * env.spec.max_episode_steps
        end_idx = (i + 1) * env.spec.max_episode_steps
        episode_data = {k: np.array(v[start_idx:end_idx]) for k, v in dataset.items()}

        # then, break up this episode if there are `done` signals
        if sum(episode_data["dones"]) > 0:
            done_idxs = np.where(episode_data["dones"])[0]
            # if there are consecutive done, only keep the last one
            last_of_consecutive_dones = np.where(np.diff(done_idxs) > 1)[0]
            if len(last_of_consecutive_dones) > 0:
                # get rid of consecutive dones
                done_idxs = np.append(
                    done_idxs[last_of_consecutive_dones], done_idxs[-1]
                )
            else:
                done_idxs = np.array([done_idxs[-1]])

            # break up the episode
            if done_idxs[-1] < env.spec.max_episode_steps - 1:
                # make sure all of the episode is included
                done_idxs = np.append(done_idxs, env.spec.max_episode_steps - 1)
            episode_start_idx = 0
            for i_done in done_idxs:
                episode_data_split = {
                    k: v[episode_start_idx : i_done + 1]
                    for k, v in episode_data.items()
                }
                episodes_dict_list.append(episode_data_split)
                episode_start_idx = i_done + 1

        else:
            # no done signal, all collection-episode is one real episode
            episodes_dict_list.append(episode_data)

    # add mc returns to each episode
    for episode_data in episodes_dict_list:
        episode_data["mc_returns"] = calc_return_to_go(
            env.spec.name,
            episode_data["rewards"],
            1 - episode_data["dones"],
            gamma,
            reward_scale,
            reward_bias,
            infinite_horizon,
        )

    return concatenate_batches(episodes_dict_list)


def make_og_bench_env_and_datasets(
    env_name,
    task_id: int = 1,
    reward_scale: Optional[float] = None,
    reward_bias: Optional[float] = None,
    seed: int = 0,
):
    """
    for OG Bench envs, turn the env into non-GC by selecting one task
    Then, use that task to create reward into the offline datasets
    """
    if "maze" in env_name:
        # OG Bench locomotion envs
        env_kwargs = dict(add_noise_to_goal=False)
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            env_name, compact_dataset=False, **env_kwargs
        )

        env = MazeSingleTaskWrapper(env, task_id)
        if reward_scale is not None and reward_bias is not None:
            env = ScaledRewardWrapper(env, scale=reward_scale, bias=reward_bias)

        # scale and clip actions
        env = gymnasium.wrappers.RescaleAction(env, -0.999, 0.999)
        env = gymnasium.wrappers.ClipAction(env)
        train_dataset["actions"] = np.clip(train_dataset["actions"], -0.999, 0.999)
        val_dataset["actions"] = np.clip(val_dataset["actions"], -0.999, 0.999)

        for ds in (train_dataset, val_dataset):
            maze_convert_to_single_task_dataset(env_name, env, ds, task_id)
            if reward_scale is not None and reward_bias is not None:
                ds["rewards"] = ds["rewards"] * reward_scale + reward_bias

    elif "cube" in env_name or "scene" in env_name or "puzzle" in env_name:
        # OG Bench manipulation envs
        # no reward scale and bias here because they are applied in the wrapper
        assert reward_scale is None or reward_scale == 1, reward_scale
        assert reward_bias is None or reward_bias == 0, reward_bias

        if "cube" in env_name or "scene" in env_name:
            env_kwargs = dict(permute_blocks=False)
        else:
            env_kwargs = dict()

        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            env_name, compact_dataset=False, **env_kwargs
        )

        env = ManipSingleTaskWrapper(env_name, env, task_id, reward_type="negative")

        for ds in (train_dataset, val_dataset):
            manip_convert_to_single_task_dataset(
                env_name, env, ds, task_id, reward_type="negative"
            )

    else:
        raise ValueError(f"Invalid OG Bench env name: {env_name}")

    return env, train_dataset, val_dataset


class ScaledRewardWrapper(gym.Wrapper):
    def __init__(self, env, scale: float, bias: float):
        super().__init__(env)
        self.scale = scale
        self.bias = bias

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward(reward)
        return ob, reward, terminated, truncated, info

    def reward(self, reward):
        return reward * self.scale + self.bias


class MazeSingleTaskWrapper(gym.Wrapper):
    def __init__(self, env, task_id=1):
        super().__init__(env)

        self.task_id = task_id

    def reset(self, options=None, *args, **kwargs):
        if options is None:
            options = {}
        options["task_id"] = self.task_id
        return self.env.reset(options=options, *args, **kwargs)


class ManipSingleTaskWrapper(gym.Wrapper):
    def __init__(self, env_name, env, task_id=1, reward_type="negative"):
        super().__init__(env)

        self.env_name = env_name
        self.task_id = task_id
        self.reward_type = reward_type

        self.prev_successes = None

    def reset(self, options=None, *args, **kwargs):
        if options is None:
            options = {}
        options["task_id"] = self.task_id
        ob, info = self.env.reset(options=options, *args, **kwargs)

        self.prev_successes = self.compute_successes()

        return ob, info

    def compute_successes(self):
        if "cube" in self.env_name:
            target_cube_xyzs = self.unwrapped._data.mocap_pos.copy()
            cur_cube_xyzs = np.array(
                [
                    self.unwrapped._data.joint(f"object_joint_{i}").qpos[:3]
                    for i in range(self.unwrapped._num_cubes)
                ]
            )
            successes = (
                np.linalg.norm(target_cube_xyzs - cur_cube_xyzs, axis=-1) <= 0.04
            ).astype(np.int64)
        elif "scene" in self.env_name:
            target_cube_xyz = self.unwrapped._data.mocap_pos[0].copy()
            target_button_states = self.unwrapped._target_button_states.copy()
            target_drawer_pos = self.unwrapped._target_drawer_pos
            target_window_pos = self.unwrapped._target_window_pos
            cur_cube_xyz = self.unwrapped._data.joint(f"object_joint_0").qpos[:3].copy()
            cur_button_states = self.unwrapped._cur_button_states.copy()
            cur_drawer_pos = self.unwrapped._data.joint("drawer_slide").qpos[0]
            cur_window_pos = self.unwrapped._data.joint("window_slide").qpos[0]
            successes = [
                np.linalg.norm(target_cube_xyz - cur_cube_xyz) <= 0.04,
                cur_button_states[0] == target_button_states[0],
                cur_button_states[1] == target_button_states[1],
                np.abs(cur_drawer_pos - target_drawer_pos) <= 0.04,
                np.abs(cur_window_pos - target_window_pos) <= 0.04,
            ]
            successes = np.array(successes, dtype=np.int64)
        elif "puzzle" in self.env_name:
            target_button_states = self.unwrapped._target_button_states.copy()
            cur_button_states = self.unwrapped._cur_button_states.copy()
            successes = []
            for i in range(self.unwrapped._num_buttons):
                successes.append(cur_button_states[i] == target_button_states[i])
            successes = np.array(successes, dtype=np.int64)
        return successes

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)

        cur_successes = self.compute_successes()

        if self.reward_type == "delta":
            reward = cur_successes.sum() - self.prev_successes.sum()
        elif self.reward_type == "cumulative":
            reward = cur_successes.sum()
        elif self.reward_type == "negative":
            reward = cur_successes.sum() - len(cur_successes)
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")

        self.prev_successes = cur_successes
        if self.reward_type in ["delta", "cumulative"]:
            if truncated:
                terminated = True  # Ground to zero if truncated.

        return ob, reward, terminated, truncated, info


def maze_convert_to_single_task_dataset(env_name, env, dataset, task_id):
    env.reset(options=dict(task_id=task_id))

    goal_xy = env.unwrapped.cur_goal_xy
    goal_tol = env.unwrapped._goal_tol

    if "maze" in env_name:
        # maze: 2D position of the agent.
        dists = np.linalg.norm(dataset["observations"][:, 0:2] - goal_xy, axis=-1)
    else:
        # antsoccer: 2D position of the ball.
        dists = np.linalg.norm(dataset["observations"][:, 15:17] - goal_xy, axis=-1)
    successes = (dists <= goal_tol).astype(np.float32)
    dataset["rewards"] = successes
    dataset["terminals"] = dataset["dones"] = successes
    dataset["masks"] = 1.0 - successes


def manip_convert_to_single_task_dataset(
    env_name, env, dataset, task_id, reward_type="negative"
):
    env.reset(options=dict(task_id=task_id))

    xyz_center = np.array([0.425, 0.0, 0.0])
    xyz_scaler = 10.0
    drawer_scaler = 18.0
    window_scaler = 15.0
    if "cube" in env_name:
        num_cubes = env.unwrapped._num_cubes
        target_cube_xyzs = env.unwrapped._data.mocap_pos.copy()
    elif "scene" in env_name:
        target_cube_xyz = env.unwrapped._data.mocap_pos[0].copy()
        target_button_states = env.unwrapped._target_button_states.copy()
        target_drawer_pos = env.unwrapped._target_drawer_pos
        target_window_pos = env.unwrapped._target_window_pos
    elif "puzzle" in env_name:
        num_buttons = env.unwrapped._num_buttons
        target_button_states = env.unwrapped._target_button_states.copy()

    observations = dataset["observations"]
    terminals = dataset["terminals"]

    @jit(nopython=True)
    def compute_rewards_and_masks():
        prev_successes = None
        rewards = np.zeros_like(observations[:, 0])
        masks = np.ones_like(observations[:, 0])

        for i in range(len(observations)):
            if "cube" in env_name:
                cur_cube_xyzs = np.zeros((num_cubes, 3))
                successes = np.zeros(num_cubes, dtype=np.int64)
                for j in range(num_cubes):
                    cur_cube_xyz = observations[i, 19 + j * 9 : 19 + j * 9 + 3]
                    cur_cube_xyz = cur_cube_xyz / xyz_scaler + xyz_center
                    cur_cube_xyzs[j] = cur_cube_xyz
                    successes[j] = int(
                        np.linalg.norm(target_cube_xyzs[j] - cur_cube_xyz) <= 0.04
                    )
            elif "scene" in env_name:
                successes = np.zeros((5,), dtype=np.int64)
                cur_cube_xyz = observations[i, 19:22]
                cur_cube_xyz = cur_cube_xyz / xyz_scaler + xyz_center
                cur_button_states = [observations[i, 29], observations[i, 33]]
                cur_drawer_pos = observations[i, 36] / drawer_scaler
                cur_window_pos = observations[i, 38] / window_scaler
                successes[0] = int(
                    np.linalg.norm(target_cube_xyz - cur_cube_xyz) <= 0.04
                )
                successes[1] = int(cur_button_states[0] == target_button_states[0])
                successes[2] = int(cur_button_states[1] == target_button_states[1])
                successes[3] = int(np.abs(cur_drawer_pos - target_drawer_pos) <= 0.04)
                successes[4] = int(np.abs(cur_window_pos - target_window_pos) <= 0.04)
            elif "puzzle" in env_name:
                successes = np.zeros((num_buttons,), dtype=np.int64)
                for j in range(num_buttons):
                    cur_button_state = observations[i, 20 + j * 4]
                    successes[j] = int(cur_button_state == target_button_states[j])
            if prev_successes is None:
                prev_successes = successes

            if reward_type == "delta":
                rewards[i] = successes.sum() - prev_successes.sum()
            elif reward_type == "cumulative":
                rewards[i] = successes.sum()
            elif reward_type == "negative":
                rewards[i] = successes.sum() - len(successes)

            masks[i] = 1.0 - np.all(successes)
            if reward_type in ["delta", "cumulative"]:
                if terminals[i] == 1.0:
                    masks[i] = 0.0  # Ground to zero at the end of the trajectory.

            if terminals[i] == 1.0:
                prev_successes = None
            else:
                prev_successes = successes

        return rewards, masks

    rewards, masks = compute_rewards_and_masks()
    dataset["rewards"] = rewards
    dataset["terminals"] = dataset["dones"] = 1 - masks
    dataset["masks"] = masks


class OGBenchSuccessWrapper(gym.Wrapper):
    """issues a 0-1 sparse reward when the success signal is given by the environment"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        success = info["success"]
        assert success in (0, 1)
        reward = success
        return ob, reward, terminated, truncated, info


if __name__ == "__main__":
    possible_envs = [
        "antmaze-large-navigate-v0",
        "humanoidmaze-medium-navigate-v0",
        # 'antsoccer-arena-navigate-v0',
        "cube-single-play-v0",
        "scene-play-v0",
        "puzzle-3x3-play-v0",
    ]

    for env_name in possible_envs:
        env, train_ds, val_ds = make_og_bench_env_and_datasets(env_name)
        env
        train_ds
        val_ds
