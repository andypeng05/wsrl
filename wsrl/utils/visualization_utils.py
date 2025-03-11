import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def make_single_trajectory_visual(
    q_estimates,
    values,
    advantages,
    rewards,
    masks,
    obs_images,
    goal_images,
    bellman_loss,
):
    def np_unstack(array, axis):
        arr = np.split(array, array.shape[axis], axis)
        arr = [a.squeeze() for a in arr]
        return arr

    def process_images(images):
        # assume image in C, H, W shape
        assert len(images.shape) == 4
        assert images.shape[-1] == 3

        interval = max(1, images.shape[0] // 4)

        sel_images = images[::interval]
        sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
        return sel_images

    fig, axs = plt.subplots(8, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    obs_images = process_images(obs_images)
    goal_images = process_images(goal_images)

    axs[0].imshow(obs_images)
    axs[1].imshow(goal_images)

    if len(q_estimates.shape) == 2:
        axs[2].plot(q_estimates[0, :], linestyle="--", marker="o")
        axs[2].plot(q_estimates[1, :], linestyle="--", marker="o")
    else:
        axs[2].plot(q_estimates, linestyle="--", marker="o")
    axs[2].set_ylabel("q values")

    if len(values.shape) == 2:
        axs[3].plot(values[0, :], linestyle="--", marker="o")
        axs[3].plot(values[1, :], linestyle="--", marker="o")
    else:
        axs[3].plot(values, linestyle="--", marker="o")
    axs[3].set_ylabel("values")

    if len(advantages.shape) == 2:
        axs[4].plot(advantages[0, :], linestyle="--")
        axs[4].plot(advantages[1, :], linestyle="--")
    else:
        axs[4].plot(advantages, linestyle="--")
    axs[4].set_ylabel("advantages")
    axs[4].set_xlim([0, advantages.shape[-1]])

    axs[5].plot(bellman_loss, linestyle="--", marker="o")
    axs[5].set_ylabel("bellman_loss")
    axs[5].set_xlim([0, bellman_loss.shape[-1]])

    axs[6].plot(rewards, linestyle="--", marker="o")
    axs[6].set_ylabel("rewards")
    axs[6].set_xlim([0, rewards.shape[-1]])

    axs[7].plot(masks, linestyle="--", marker="o")
    axs[7].set_ylabel("masks")
    axs[7].set_xlim([0, masks.shape[-1]])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    return out_image


def value_and_reward_visulization(trajs, agent, discount=0.99, seed=0):
    def _batch_dicts(stat):
        """stat is a list of dict, turn it into a dict of list"""
        d = {}
        for k in stat[0].keys():
            d[k] = np.array([s[k] for s in stat])
        return d

    rng = jax.random.PRNGKey(seed)
    n_trajs = len(trajs)
    visualization_images = []

    # for each trajectory
    for i in range(n_trajs):
        if type(trajs[i]["observations"]) == list:
            # obs is [{'image': ..., 'proprio': ...},  {}]
            # as returned by evaluate() funcs
            observations = _batch_dicts(trajs[i]["observations"])
            next_observations = _batch_dicts(trajs[i]["next_observations"])
            goals = _batch_dicts(trajs[i]["goals"])
        else:
            # obs is {'image': Array(...), 'proprio': Array(...)}
            # as returned by dataset iterators
            observations = trajs[i]["observations"]
            next_observations = trajs[i]["next_observations"]
            goals = trajs[i]["goals"]
        actions = np.array(trajs[i]["actions"])
        rewards = np.array(trajs[i]["rewards"])
        if "masks" in trajs[i]:
            masks = np.array(trajs[i]["masks"])
        else:
            masks = np.array([not d for d in trajs[i]["dones"]])

        q_pred = agent.forward_critic(
            (observations, goals), actions, rng=None, train=False
        )

        # check if agent has method forward_value
        if hasattr(agent, "forward_value"):
            # IQL agent
            values = agent.forward_value((observations, goals), rng, train=False)
        else:
            # CQL agent
            values = agent.forward_critic(
                (observations, goals),
                agent.forward_policy((observations, goals), rng, train=False).sample(
                    seed=rng
                ),
                rng=None,
                train=False,
            )

        advantages = q_pred - values

        next_actions_dist = agent.forward_policy(
            (next_observations, goals), rng, train=False
        )
        next_actions = next_actions_dist.sample(seed=rng)
        target_q_pred = agent.forward_critic(
            (next_observations, goals), next_actions, rng=None, train=False
        )
        if len(target_q_pred.shape) == 2:
            # min over the ensemble dim
            target_q_pred = target_q_pred.min(axis=0)
        td_target = rewards + target_q_pred * discount * masks
        td_loss = (q_pred - td_target) ** 2
        if len(td_loss.shape) == 2:
            td_loss = td_loss.mean(axis=0)

        visualization_images.append(
            make_single_trajectory_visual(
                q_pred,
                values,
                advantages,
                rewards,
                masks,
                next_observations["image"],
                goals["image"],
                td_loss,
            )
        )

    return np.concatenate(visualization_images, 0)