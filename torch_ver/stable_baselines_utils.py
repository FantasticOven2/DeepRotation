from functools import partial
from typing import Tuple, Dict

import gym
import numpy as np
import torch
from stable_baselines3.common.distributions import \
    Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor as SACActor, MlpPolicy
from torch import nn, Tensor

from tsGaussian import torch_tsgaussian, utils
from models import PointNet, LatentNet

class TangentSpaceGaussian(Distribution):

    def __init__(self, device: torch.device):
        super(TangentSpaceGaussian, self).__init__()
        self.distribution = torch_tsgaussian.TangentSpaceGaussian(device)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0
                                ) -> Tuple[nn.Module, nn.Parameter]:
        raise NotImplementedError('Not needed.')

    def proba_distribution(self, mu: Tensor, sigma: Tensor) -> 'TangentSpaceGaussian':
        self.mu = mu
        self.sigma = sigma
        return self

    def log_prob(self, actions) -> Tensor:
        log_prob = self.distribution.log_probs(actions, self.mu, self.sigma) # Parameter for log_probs? R_mu, R_x?
        return log_prob

    def entropy(self) -> Tensor:
        return self.distribution.entropy(self.mu, self.sigma)

    def sample(self) -> Tensor:
        """Need to be implemented"""
        s = self.distribution.rsample(self.mu, self.sigma)
        return s

    """What is this mode function"""
    def mode(self) -> Tensor:
        return self.mu

    """Need to be clear with these functions parameters"""
    def actions_from_params(self, mu: Tensor, sigma: Tensor,
                            deterministic: bool = False) -> Tensor:
        self.proba_distribution(mu, sigma)
        # actions = self.get_actions(deterministic = deterministic)
        # print('actions size: ', actions.size())
        # print('actions type: ', type(actions))
        return self.get_actions(deterministic = deterministic)

    def log_prob_from_params(self, mu, sigma):
        actions, log_prob = self.actions_from_params(mu, sigma)
        # print('actions: ', actions)
        # print('actions_mat: ', actions_mat)
        # log_prob = self.log_prob(actions)
        # print('log prob: ', log_prob[0])
        # print('prob: ', np.e ** log_prob[0])
        return actions, log_prob

class CustomSACActor(SACActor):

    def __init__(self, *args, **kwargs):
        super(CustomSACActor, self).__init__(*args, **kwargs)
        assert not self.use_sde
        self.log_std = self.mu = None
        del self.log_std
        del self.mu
        last_layer_dim = self.net_arch[-1] if len(
            self.net_arch) > 0 else self.features_dim

        ### Quaternion Representation ###    
        # self.vec7 = nn.Linear(last_layer_dim, 7)
        # self.vec3 = nn.Linear(last_layer_dim, 3)
        self.action_dist = None
        self.latent_pi = LatentNet()

    def get_std(self) -> Tensor:
        raise NotImplementedError('Not needed.')

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotImplementedError('Not needed.')

    def get_action_dist_params(self, obs: Tensor) -> Tuple[
        Tensor, Tensor, Dict[str, Tensor]]:
        if self.action_dist is None:
            self.action_dist = TangentSpaceGaussian(self.device)
        # print('obs: ', obs)
        ''' Print out features, latent_pi, vec12 dims, see change in matrix L2 norm (Largest singular value) / Forbenius norm'''
        features = self.extract_features(obs)
        vec7 = self.latent_pi(features)
        print('Features: ', torch.norm(features))
        print('vec7: ', torch.norm(vec7))
        # import IPython
        # IPython.embed()
        mu, sigma = utils.vec9_to_mu_sigma(vec7)
        if vec7[0][0] >= 20:
            raise Exception("Exploding")
        return mu, sigma, {}

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        mu, sigma, kwargs = self.get_action_dist_params(obs)
        action = self.action_dist.actions_from_params(mu, sigma,
                                                    deterministic = deterministic,
                                                    **kwargs)
        return action

    def action_log_prob(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        mu, sigma, kwargs = self.get_action_dist_params(obs)
        a, lp = self.action_dist.log_prob_from_params(mu, sigma, **kwargs)
        return a, lp

    def _predict(self, observation: Tensor,
                    deterministic: bool = False) -> Tensor:
        return self.forward(observation, deterministic)

class CustomSACPolicy(MlpPolicy):
    def make_actor(self, features_extractor = None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor)
        return CustomSACActor(**actor_kwargs).to(self.device)

class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        assert not self.use_sde

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = None
        self.action_net = nn.Linear(latent_dim_pi, 7)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(),
                                              lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[
        Tensor, Tensor, Tensor]:
        if self.action_dist is None:
            self.action_dist = TangentSpaceGaussian(self.device)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        # print(actions.shape)
        # print('actions size: ', actions.shape)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: Tensor) -> Distribution:
        vec12 = self.action_net(latent_pi)
        mu, sigma = utils.vec7_to_mu_sigma(vec12)
        return self.action_dist.proba_distribution(mu, sigma)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self._model = PointNet(features_dim, batchnorm=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        pred = self._model(observations)
        return pred
