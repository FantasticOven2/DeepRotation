import torch
from absl import app
from absl import flags
from stable_baselines3 import SAC, PPO
from torch import autograd


from envs.wahba import Wahba
from stable_baselines_utils import CustomSACPolicy, \
    CustomCNN, CustomActorCriticPolicy

FLAGS = flags.FLAGS

flags.DEFINE_enum('alg', 'sac_bpp', ['ppo', 'sac', 'sac_bpp', 'ppo_bpp'],
                  'Algorithm to run.')


def main(argv):
    env = Wahba()
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    print(FLAGS.alg)
    # if 'sac' in FLAGS.alg:
    #     policy_kwargs['n_critics'] = 1
    #     policy_kwargs['share_features_extractor'] = False
    #     policy = 'MlpPolicy' if FLAGS.alg == 'sac' else CustomSACPolicy
    #     # policy = CustomSACPolicy
    #     # policy = 'MlpPolicy'
    #     # print(policy)
    #     model = SAC(policy, env, verbose=1, ent_coef='auto_0.1',
    #                 policy_kwargs=policy_kwargs, device=device)
    # else:
    #     print('PPO')
    #     policy = 'MlpPolicy' if FLAGS.alg == 'ppo' else CustomActorCriticPolicy
    #     # policy = CustomActorCriticPolicy
    #     model = PPO(policy, env, verbose=1, policy_kwargs=policy_kwargs, ent_coef='auto_0.1',
                # device=device)
    policy_kwargs['n_critics'] = 1
    policy_kwargs['share_features_extractor'] = False
    policy = 'MlpPolicy' if FLAGS.alg == 'sac' else CustomSACPolicy
    # policy = 'MlpPolicy'
    model = SAC(policy, env, verbose=1, ent_coef=0.1,
                policy_kwargs=policy_kwargs, device=device)
    # print('PPO')
    # policy = 'MlpPolicy' if FLAGS.alg == 'ppo' else CustomActorCriticPolicy
    # model = PPO(policy, env, verbose=1, policy_kwargs=policy_kwargs, ent_coef=0.1, 
    #             device=device)
    model.learn(total_timesteps=100000, eval_freq=100, n_eval_episodes=100) # Change 500000 to 100000


if __name__ == '__main__':
    with autograd.detect_anomaly():
        app.run(main)
