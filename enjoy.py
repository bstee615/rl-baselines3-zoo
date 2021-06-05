import argparse
import importlib
import os
import random
import sys

import numpy as np
import torch as th
import pandas as pd
import seaborn as sns
import yaml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from len_gen import BatchTqdm
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
from surprise_adequacy.surprise.surprise_adequacy import SurpriseAdequacyConfig, LSA


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
    # if isinstance(obs, np.ndarray):
    #     obs = torch.from_numpy(obs)
    obs = torch.tensor(obs).float()
    for batch in obs.split(1024):
        batch = batch.cuda()
        fgsm = GradientSignAttack(
            predict=predict, eps=eps*255,
            clip_min=0, clip_max=255, targeted=True)
        original_action, _, _ = net(batch)

        adv_batch = fgsm.perturb(batch, original_action)
        adv_action, _, _ = net(adv_batch)

        attack_succeeded.append((~original_action.eq(adv_action)).cpu().numpy())
        adv_obs.append(adv_batch.cpu().int().numpy())
    # plt.imshow(np.concatenate(batch[0].cpu().int().numpy()), cmap='gray')
    # plt.title(f'benign')
    # plt.show()
    # plt.imshow(np.concatenate(adv_batch[0].cpu().int().numpy()), cmap='gray')
    # plt.title(f'$\\epsilon = {eps}$')
    # plt.show()
    attack_succeeded = np.concatenate(attack_succeeded)
    adv_obs = np.concatenate(adv_obs)

    return adv_obs, attack_succeeded


def do_surprise(model, sa, log_dir, layer, eps):
    log_dir = Path(log_dir)
    lsa_dir = log_dir / 'lsa'
    lsa_dir.mkdir(exist_ok=True)
    if eps is None:
        target = 'test'
    else:
        target = f'fgsm_{eps}'

    sa.config = SurpriseAdequacyConfig(saved_path=lsa_dir, is_classification=True, layer_names=[layer],
                                       ds_name='pong', num_classes=6)
    # if Path(train_dir / 'ats.npy').exists():
    #     train_ats = np.load(str(train_dir / 'ats.npy'))
    #     train_preds = np.load(str(train_dir / 'preds.npy'))
    # else:
    #     train_obs, num_train_obs = SaveObservationCallback.load(obs_pkl)
    #     train_activations, train_preds = get_activations(model.policy, [layer], train_obs, num_train_obs)
    #     np.save(str(train_dir / 'activations.npy'), train_activations)
    #     np.save(str(train_dir / 'preds.npy'), train_preds)
    #     train_ats = process_ats(train_activations, num_proc=3)
    #     np.save(str(train_dir / 'ats.npy'), train_ats)

    if Path(lsa_dir / 'test_obs.npy').exists():
        test_obs = np.load(str(lsa_dir / 'test_obs.npy'))
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
        print(f'Mean reward: {np.mean(episode_rewards):.2f}+-{np.std(episode_rewards):.2f}')
        test_obs = np.concatenate(test_obs)
        np.save(str(lsa_dir / 'test_obs.npy'), test_obs)

    if eps is None:
        target_obs = test_obs
    else:
        adv_obs, adv_success = attack(model, test_obs, eps)
        print(f'Epsilon: {eps}, attacks succeeded: {np.sum(adv_success) / len(adv_success) * 100:.2f}%')
        target_obs = adv_obs

    # if Path(target_dir / f'ats.npy').exists():
    #     target_preds = np.load(str(target_dir / f'preds.npy'))
    #     target_ats = np.load(str(target_dir / f'ats.npy'))
    # else:
    #     num_target_obs = len(target_obs)
    #     target_obs = np.array_split(target_obs, len(target_obs) // 256)
    #     target_activations, target_preds = get_activations(model.policy, [layer], target_obs, num_target_obs)
    #     np.save(str(target_dir / f'activations.npy'), target_activations)
    #     np.save(str(target_dir / f'preds.npy'), target_preds)
    #     target_ats = process_ats(target_activations)
    #     np.save(str(target_dir / f'ats.npy'), target_ats)
    if not Path(lsa_dir / f'lsa_{layer}_{target}.npy').exists():
        sa.prep(use_cache=True)
        lsa, pred = sa.calc(
            BatchTqdm(iter(np.array_split(target_obs, len(target_obs) // 256)), len(target_obs)),
            ds_type=target,
            use_cache=True
        )

        np.save(str(lsa_dir / f'lsa_{layer}_{target}.npy'), lsa)
    else:
        lsa = np.load(str(lsa_dir / f'lsa_{layer}_{target}.npy'))

    # if Path(target_dir / f'lsa.npy').exists():
    #     lsa = np.load(str(target_dir / f'lsa.npy'))
    # else:
    #     my_args = SimpleNamespace()
    #     my_args.var_threshold = 1e-5
    #     my_args.num_classes = model.env.action_space.n
    #     my_args.is_classification = True
    #     # lsa = do_lsa(my_args, train_ats, train_preds, target_ats, target_preds)
    #     np.save(str(save_dir / f'{target}_lsa.npy'), lsa)
    # print(train_ats.shape, target_ats.shape, len(lsa))

    plot_it(log_dir / 'plots', layer, eps, target, lsa)


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
    parser.add_argument(
        "--surprise", action="store_true", default=False, help="Evaluate surprise"
    )
    parser.add_argument(
        "--attack", action="store_true", default=False, help="Do adversarial attack"
    )
    parser.add_argument(
        "--sequence", action="store_true", default=False, help="Evaluate sequence of surprise"
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

    ts_to_load = None
    if args.load_checkpoint:
        # TODO: Set to sampling rate
        ts_to_load = args.load_checkpoint // 16

    log_dir = Path(log_path)
    if args.surprise:
        obs_pkl = log_dir / 'train' / 'obs.npy'
        train_obs = SaveObservationCallback.load(obs_pkl)
        all_layers = ['features_extractor.cnn.3', 'features_extractor.cnn.5', 'features_extractor.linear.1']
        # all_layers = ['features_extractor.cnn.5', 'features_extractor.linear.1']
        # all_layers = ['features_extractor.linear.1']

        # epsilons = [None, 0.01, 0.1, 0.5, 1.0]
        epsilons = [0.001, 0.005]

        for layer in all_layers:
            sa_config = SurpriseAdequacyConfig(saved_path=log_dir / 'lsa', is_classification=True, layer_names=[layer],
                                               ds_name='pong', num_classes=6)
            sa = LSA(model.policy, train_obs, sa_config)
            sa.prep(use_cache=True)
            for eps in epsilons:
            # for eps in [0.5]:
                    do_surprise(model, sa, log_path, layer, eps)
    if args.attack:
        def rollout(model, env, n_obs, eps, prob_attack, render=False):
            obs_collected = 0
            obs = env.reset()
            pbar = tqdm(total=n_obs)
            episode_rewards = []
            while obs_collected < n_obs:
                if render:
                    env.render()
                if eps is not None:
                    if random.random() < prob_attack:
                        adv_obs, success = attack(model, np.transpose(obs, (0, 3, 1, 2)), eps)
                        adv_obs = np.transpose(adv_obs, (0, 2, 3, 1))
                        obs = adv_obs
                action, state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_rewards += [i["episode"]["r"] for i in info if "episode" in i]
                obs_collected += len(obs)
                pbar.update(len(obs))
            pbar.close()
            return episode_rewards
        all_rewards = []
        # epsilons = [None, 0.01, 0.1, 0.2, 0.4, 0.8, 1.0]
        epsilons = [0.001, 0.01]
        # probs = [0.1, 0.25, 0.5, 1.0]
        probs = [1.0]
        for eps in epsilons:
            df = pd.DataFrame()
            for p in probs:
                print(f'Attacking the agent. Epsilon: {eps}')
                n_obs = 1e4
                episode_rewards = rollout(model, env, n_obs, eps, p)
                print(f'Mean reward: {np.mean(episode_rewards):.2f}+-{np.std(episode_rewards):.2f}')
                results = pd.DataFrame([[p, r] for r in episode_rewards], columns=['prob', 'rew'])
                all_rewards.append(episode_rewards)
                df = df.append(results, ignore_index=True)
                model.env.close()
            df.to_csv(log_dir / f'adv_attacks_{eps}.csv')
            sns.barplot(data=df, x='prob', y='rew', ci='sd')
            plt.title(f'Pong adversarial attack performance, $\\epsilon = {eps}$')
            plt.show()
    if args.sequence:
        env = model.env

        layer = 'features_extractor.linear.1'
        obs_pkl = log_dir / 'train' / 'obs.npy'
        train_obs = SaveObservationCallback.load(obs_pkl, ts_to_load)
        sa_config = SurpriseAdequacyConfig(saved_path=log_dir / 'lsa', is_classification=True, layer_names=[layer],
                                           ds_name='pong', num_classes=6)
        sa = LSA(model.policy, train_obs, sa_config)
        sa.prep(use_cache=True)

        # perturb_probs = [0.01, 0.1, 0.5]
        # perturb_probs = [1.0]
        perturb_probs = [0.0]
        for perturb_prob in perturb_probs:
            epsilon = 0.1
            # n_episodes = 1
            n_episodes = 100

            rewards = []
            all_eps_obs = []
            all_eps_success, all_eps_fail = [], []
            for i in range(n_episodes):
                eps_obs = []
                eps_success, eps_fail = [], []
                eps_reward = 0
                obs = env.reset()
                done = False
                ts = 0
                while not done:
                    if random.random() < perturb_prob:
                        # adv_obs, success = attack(model, np.transpose(obs, (0, 3, 1, 2)), epsilon)
                        # adv_obs = np.transpose(adv_obs, (0, 2, 3, 1))
                        adv_obs, success = attack(model, obs, epsilon)
                        succeeded = success[0]
                        if succeeded:
                            eps_success.append(ts)
                            obs = adv_obs
                        # else:
                        #     eps_fail.append(ts)
                    eps_obs.append(obs)
                    action, state = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    eps_reward += reward.item()
                    ts += 1
                    # env.render()
                eps_obs = np.concatenate(eps_obs)
                all_eps_obs.append(eps_obs)
                all_eps_success.append(eps_success)
                all_eps_fail.append(eps_fail)
                rewards.append(eps_reward)
            all_eps_success = np.array(all_eps_success)
            # all_eps_fail = np.array(all_eps_fail)
            rewards = np.array(rewards)

            variances = []
            weird_idx_lens = []

            eps_lsa = []
            scaler = StandardScaler()
            for (i, eps_obs) in enumerate(all_eps_obs):
                # num_splits = len(eps_obs) // 256
                # if num_splits == 0:
                #     num_splits = 1
                # batches = np.array_split(eps_obs, num_splits)
                batches = [eps_obs]
                lsa, pred = sa.calc(
                    BatchTqdm(
                        iter(batches),
                        len(eps_obs)
                    ), ds_type=f'sequence_fgsm_{perturb_prob}_{epsilon}', use_cache=False)
                eps_lsa.append(scaler.fit_transform(lsa.reshape((-1, 1))))

                # Print variance stats
                variance = np.var(lsa)
                variances.append(variance)
                # print('Variance:', variance)
                from scipy import stats
                weird_pts = np.abs(lsa - stats.mode(lsa).mode) > 0.0001
                weird_idx = np.where(weird_pts)[0]
                weird_idx_lens.append(len(weird_idx))
                # print('Number of weird points:', len(weird_idx), 'difference', lsa[weird_idx] - stats.mode(lsa).mode)

            print('Variance:', np.mean(variances), np.std(variances))
            print('Number of weird points:', np.mean(weird_idx_lens), np.std(weird_idx_lens))

            # return  # TODO skipping plot for now

            # df = pd.DataFrame(data=[(i, eps, lsa) for i, (eps, lsa) in enumerate(eps_lsa)], columns=["ts", "episode", "lsa"])
            # df = pd.melt(df, ['ts'])
            # sns.lineplot(data=df, x='ts', y='value', hue='variable')
            # np.save(f'lsa-seq-{args.env}.npy', eps_lsa)
            # np.save(f'rewards-seq-{args.env}.npy', rewards)
            # np.save(f'all_eps_success-seq-{args.env}.npy', all_eps_success)
            # np.save(f'all_eps_fail-seq-{args.env}.npy', all_eps_fail)
            for i, el in enumerate(eps_lsa):
                plt.plot(range(len(el)), el, label=f'episode {i}/reward {rewards[i]}')
                plt.scatter(all_eps_success[i], np.take(el.astype(np.float), all_eps_success[i].astype(np.int)), marker='o', c='g')
                # plt.scatter(all_eps_fail[i], np.take(el, all_eps_fail[i]), marker='x', c='r')
            plt.legend(loc="upper left")
            plt.xlabel('timestep')
            plt.ylabel('lsa')
            plt.title(f'{n_episodes} LSA sequences on {args.env}. $p = {perturb_prob}$, $\\epsilon = {epsilon}$')
            plt.show()


def detector(benign_lsa, fgsm_lsa):
    x = np.concatenate((benign_lsa, fgsm_lsa)).reshape((-1, 1))
    y = np.concatenate((np.zeros_like(benign_lsa), np.ones_like(fgsm_lsa)))
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # plt.scatter(np.array([x for i, x in enumerate(x_train) if y_train[i] == 1]), y_train[y_train == 1], c='b', marker='o')
    # plt.scatter(np.array([x for i, x in enumerate(x_train) if y_train[i] == 0]), y_train[y_train == 0], c='r', marker='+')
    # plt.show()

    from sklearn.linear_model import LogisticRegressionCV
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=10))
    # model = LogisticRegressionCV(cv=10)
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                       hidden_layer_sizes=(16, 16), random_state=1,
    #                       max_iter=1e4)
    trained_model = model.fit(x_train, y_train)

    pred = trained_model.predict(x_test)
    tp = np.sum(np.logical_and(pred, y_test)).item()
    fp = np.sum(np.logical_and(pred, np.logical_not(y_test))).item()
    tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(y_test))).item()
    fn = np.sum(np.logical_and(np.logical_not(pred), y_test)).item()
    return tp, fp, tn, fn


def test_detector():
    entries = []
    # layer = 'features_extractor.cnn.3'

    for game in ['Pong', 'SpaceInvaders']:
        for layer in ['features_extractor.cnn.3', 'features_extractor.cnn.5', 'features_extractor.linear.1']:
            for eps in [0.001, 0.005, 0.01, 0.1, 0.5, 1.0]:
                # game = 'Pong'
                # game = 'SpaceInvaders'
                root = Path(f'D:\\weile-lab\\sadl-rl\\runs\\pronto\\results\\zoo\\ppo\\{game}NoFrameskip-v4_1\\lsa')
                benign_lsa = np.load(root / f'lsa_{layer}_test.npy')
                fgsm_lsa = np.load(root / f'lsa_{layer}_fgsm_{eps}.npy')
                stats = detector(benign_lsa, fgsm_lsa)
                # print(f'Accuracy of LR adv detector: {accuracy * 100:.2f}%')
                entries.append((layer, eps, *stats))

        df = pd.DataFrame(entries, columns=['layer', 'eps', 'tp', 'fp', 'tn', 'fn'])
        df["precision"] = df["tp"] / (df["tp"] + df["fp"])
        df["recall"] = df["tp"] / (df["tp"] + df["fn"])

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 6)
        fig.suptitle(f'Precision/Recall of LR adv. detector over $\\epsilon$ - {game}NoFrameskip')
        sns.barplot(data=df, x='eps', y='precision', hue='layer', ci=None, ax=ax[0])
        sns.barplot(data=df, x='eps', y='recall', hue='layer', ci=None, ax=ax[1])
        ax[0].set_title('Precision')
        ax[0].set_ylim(0.0, 1.1)
        ax[1].set_title('Recall')
        ax[1].set_ylim(0.0, 1.1)

        for a in ax:
            a.legend([], [], frameon=False)
        # handles, labels = ax[0].get_legend_handles_labels()
        ax[1].legend(bbox_to_anchor=(1.03, 1))
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
        # plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
