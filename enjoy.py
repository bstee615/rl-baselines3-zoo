import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


from matplotlib import pyplot as plt
from pathlib import Path
from types import SimpleNamespace
import torch
from advertorch.attacks import GradientSignAttack
from tqdm import tqdm
from rl.callbacks_and_wrappers import SaveObservationCallback
from rl.eval import get_activations, do_lsa, plot_it
from sa import process_ats


def attack(model, obs, eps):
    net = model.policy

    def predict(x):
        """
        Predict action for observation, returning the logits which were given to the action distribution.
        Taken from stable_baselines3.common.policies.ActorCriticPolicy.forward
        """
        latent_pi, _, _ = net._get_latent(x)
        logits = net.action_net(latent_pi)
        return logits

    attack_succeeded = []
    adv_obs = []
    obs = torch.tensor(obs).float()
    for batch in obs.split(1024):
        batch = batch.cuda()
        fgsm = GradientSignAttack(
            predict=predict, eps=eps*4,
            clip_min=0, clip_max=255, targeted=True)
        original_action, _, _ = net(batch)

        adv_batch = fgsm.perturb(batch, original_action)
        adv_action, _, _ = net(adv_batch)

        attack_succeeded.append(original_action.eq(adv_action).cpu().numpy())
        adv_obs.append(adv_batch.cpu().int().numpy())
    plt.imshow(np.concatenate(batch[0].cpu().int().numpy()), cmap='gray')
    plt.show()
    plt.imshow(np.concatenate(adv_batch[0].cpu().int().numpy()), cmap='gray')
    plt.show()
    attack_succeeded = np.concatenate(attack_succeeded)
    adv_obs = np.concatenate(adv_obs)

    return adv_obs, attack_succeeded


def do_surprise(model, log_dir):
    log_dir = Path(log_dir)
    obs_pkl = log_dir / 'train' / 'obs.npy'
    save_dir = log_dir / 'save'
    layers = ['cnn.3']
    if Path(save_dir / 'train_ats.npy').exists():
        train_ats = np.load(str(save_dir / 'train_ats.npy'))
        train_preds = np.load(str(save_dir / 'train_preds.npy'))
    else:
        train_obs, num_train_obs = SaveObservationCallback.load(obs_pkl)
        train_activations, train_preds = get_activations(model.policy, layers, train_obs, num_train_obs)
        np.save(str(save_dir / 'train_activations.npy'), train_activations)
        np.save(str(save_dir / 'train_preds.npy'), train_preds)
        train_ats = process_ats(train_activations, num_proc=3)
        np.save(str(save_dir / 'train_ats.npy'), train_ats)

    if Path(save_dir / 'test_obs.npy').exists():
        test_obs = np.load(str(save_dir / 'test_obs.npy'))
    else:
        env = model.env
        print('Gathering observations from environment')
        n_obs = 1e4
        pbar = tqdm(total=n_obs)
        obs_collected = 0
        episode_rewards = []
        test_obs = []
        obs = env.reset()
        while obs_collected < n_obs:
            # obs = uniform_attack(cfg, obs, model)
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards += [i["episode"]["r"] for i in info if "episode" in i]
            test_obs.append(obs)
            obs_collected += len(obs)
            pbar.update(len(obs))
        pbar.close()
        print(f'Mean reward: {np.mean(episode_rewards):.2f}+={np.std(episode_rewards)}:.2f')
        test_obs = np.concatenate(test_obs)
        np.save(str(save_dir / 'test_obs.npy'), test_obs)

    eps = 1.0
    # target = f'fgsm_{eps}'
    target = 'test'
    adv_obs, adv_success = attack(model, test_obs, eps)
    print(f'Epsilon: {eps}, attacks succeeded: {np.sum(adv_success) / len(adv_success) * 100:.2f}%')
    if 'fgsm' in target:
        target_obs = adv_obs
    else:
        target_obs = test_obs

    if Path(save_dir / f'{target}_ats.npy').exists():
        target_preds = np.load(str(save_dir / f'{target}_preds.npy'))
        target_ats = np.load(str(save_dir / f'{target}_ats.npy'))
    else:
        num_target_obs = len(target_obs)
        target_obs = np.array_split(target_obs, len(target_obs) // 256)
        target_activations, target_preds = get_activations(model.policy, layers, target_obs, num_target_obs)
        np.save(str(save_dir / f'{target}_activations.npy'), target_activations)
        np.save(str(save_dir / f'{target}_preds.npy'), target_preds)
        target_ats = process_ats(target_activations)
        np.save(str(save_dir / f'{target}_ats.npy'), target_ats)

    if Path(save_dir / f'{target}_lsa.npy').exists():
        lsa = np.load(str(save_dir / f'{target}_lsa.npy'))
    else:
        my_args = SimpleNamespace()
        my_args.var_threshold = 1e-5
        my_args.num_classes = model.env.action_space.n
        my_args.is_classification = True
        lsa = do_lsa(my_args, train_ats, train_preds, target_ats, target_preds)
        np.save(str(save_dir / f'{target}_lsa.npy'), lsa)
    print(train_ats.shape, target_ats.shape, len(lsa))

    plot_it(log_dir / 'plots', layers, eps, target, lsa)


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    do_surprise(model, log_path)

    # obs = env.reset()
    #
    # # Deterministic by default except for atari games
    # stochastic = args.stochastic or is_atari and not args.deterministic
    # deterministic = not stochastic
    #
    # state = None
    # episode_reward = 0.0
    # episode_rewards, episode_lengths = [], []
    # ep_len = 0
    # # For HER, monitor success rate
    # successes = []
    # try:
    #     for _ in range(args.n_timesteps):
    #         action, state = model.predict(obs, state=state, deterministic=deterministic)
    #         obs, reward, done, infos = env.step(action)
    #         if not args.no_render:
    #             env.render("human")
    #
    #         episode_reward += reward[0]
    #         ep_len += 1
    #
    #         if args.n_envs == 1:
    #             # For atari the return reward is not the atari score
    #             # so we have to get it from the infos dict
    #             if is_atari and infos is not None and args.verbose >= 1:
    #                 episode_infos = infos[0].get("episode")
    #                 if episode_infos is not None:
    #                     print(f"Atari Episode Score: {episode_infos['r']:.2f}")
    #                     print("Atari Episode Length", episode_infos["l"])
    #
    #             if done and not is_atari and args.verbose > 0:
    #                 # NOTE: for env using VecNormalize, the mean reward
    #                 # is a normalized reward when `--norm_reward` flag is passed
    #                 print(f"Episode Reward: {episode_reward:.2f}")
    #                 print("Episode Length", ep_len)
    #                 episode_rewards.append(episode_reward)
    #                 episode_lengths.append(ep_len)
    #                 episode_reward = 0.0
    #                 ep_len = 0
    #                 state = None
    #
    #             # Reset also when the goal is achieved when using HER
    #             if done and infos[0].get("is_success") is not None:
    #                 if args.verbose > 1:
    #                     print("Success?", infos[0].get("is_success", False))
    #
    #                 if infos[0].get("is_success") is not None:
    #                     successes.append(infos[0].get("is_success", False))
    #                     episode_reward, ep_len = 0.0, 0
    #
    # except KeyboardInterrupt:
    #     pass
    #
    # if args.verbose > 0 and len(successes) > 0:
    #     print(f"Success rate: {100 * np.mean(successes):.2f}%")
    #
    # if args.verbose > 0 and len(episode_rewards) > 0:
    #     print(f"{len(episode_rewards)} Episodes")
    #     print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    #
    # if args.verbose > 0 and len(episode_lengths) > 0:
    #     print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    #
    # env.close()


if __name__ == "__main__":
    main()
