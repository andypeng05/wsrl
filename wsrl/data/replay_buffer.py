import os
from typing import Iterable, Optional, Union

import gym
import gym.spaces
import gymnasium
import jax
import numpy as np
from absl import flags

from wsrl.data.dataset import Dataset, DatasetDict, _sample
from wsrl.envs.env_common import calc_return_to_go


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box) or isinstance(
        obs_space, gymnasium.spaces.Box
    ):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict) or isinstance(
        obs_space, gymnasium.spaces.Dict
    ):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert (
            dataset_dict.keys() == data_dict.keys()
        ), f"{dataset_dict.keys()} != {data_dict.keys()}"
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        dataset_dict: DatasetDict,
        capacity: int,
        seed: Optional[int] = None,
        discount: Optional[int] = None,
    ):
        super().__init__(dataset_dict, seed)
        self._discount = discount
        self._capacity = capacity

    @classmethod
    def create_new(
        cls,
        capacity: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
            dones=np.empty((capacity,), dtype=np.float32),
        )

        replay_buffer = cls(dataset_dict, capacity, seed, discount)
        replay_buffer._size = 0
        replay_buffer._insert_index = 0
        replay_buffer._unsampled_indices = None  # for now, only use this when we load
        return replay_buffer

    @classmethod
    def create_from_initial_dataset(
        cls,
        init_dataset: dict,
        capacity: int,
        seed: Optional[int] = None,
        discount: Optional[int] = None,
    ):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            capacity: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros(
                (capacity, *init_buffer.shape[1:]), dtype=init_buffer.dtype
            )
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        replay_buffer = cls(buffer_dict, capacity, seed, discount)
        replay_buffer._size = replay_buffer._insert_index = get_size(init_dataset)

        replay_buffer._unsampled_indices = None  # for now, only use this when we load
        return replay_buffer

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample_without_repeat(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
    ) -> dict:
        if keys is None:
            keys = self.dataset_dict.keys()

        batch = dict()
        if len(self.unsampled_indices) < batch_size:
            raise ValueError("Not enough samples left to sample without repeat.")
        selected_indices = []
        for _ in range(batch_size):
            idx = self.np_random.randint(len(self.unsampled_indices))
            selected_indices.append(self.unsampled_indices[idx])
            # Swap the selected index with the last unselected index
            self.unsampled_indices[idx], self.unsampled_indices[-1] = (
                self.unsampled_indices[-1],
                self.unsampled_indices[idx],
            )
            # Remove the last unselected index (which is now the selected index)
            self.unsampled_indices.pop()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], np.array(selected_indices))

        return batch

    def save(self, save_dir):
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        np.save(save_buffer_file, self.dataset_dict)
        np.save(save_size_file, self._size)

    def load(self, save_dir):
        # TODO: maybe make sure the dataset_dict thats being loaded has mc_returns if self is ReplayBufferMC
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        self.dataset_dict = np.load(save_buffer_file, allow_pickle=True).item()
        self._size = np.load(save_size_file, allow_pickle=True).item()
        self.unsampled_indices = list(range(self._size))


class ReplayBufferMC(ReplayBuffer):
    def __init__(
        self,
        dataset_dict: DatasetDict,
        capacity: int,
        seed: Optional[int] = None,
        discount: Optional[int] = None,
    ):
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super.__init__(dataset_dict, capacity, seed, discount)
        mc_returns = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["mc_returns"] = mc_returns

        self._allow_idxs = []
        self._traj_start_idx = 0

    def insert(self, data_dict: DatasetDict):
        # assumes replay buffer capacity is more than the number of online steps
        assert self._size < self._capacity, "replay buffer has reached capacity"

        data_dict["mc_returns"] = None
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        # if "dones" not in data_dict:
        #     data_dict["dones"] = 1 - data_dict["masks"]

        if data_dict["dones"] == 1.0:
            # compute the mc_returns
            FLAGS = flags.FLAGS
            rewards = self.dataset_dict["rewards"][
                self._traj_start_idx : self._insert_index + 1
            ]
            masks = self.dataset_dict["masks"][
                self._traj_start_idx : self._insert_index + 1
            ]
            self.dataset_dict["mc_returns"][
                self._traj_start_idx : self._insert_index + 1
            ] = calc_return_to_go(
                FLAGS.env,
                rewards,
                masks,
                self._discount,
            )

            self._allow_idxs.extend(
                list(range(self._traj_start_idx, self._insert_index + 1))
            )
            self._traj_start_idx = self._insert_index + 1

        self._size += 1
        self._insert_index += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        if indx is None:
            indx = self.np_random.choice(
                self._allow_idxs, size=batch_size, replace=True
            )
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, indx, side="right") - 1
            ]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(indx - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[indx], self["next_observations"])
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch
