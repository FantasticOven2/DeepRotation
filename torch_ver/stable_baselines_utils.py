import torch
import gym
from typing import Tuple, Dict
from stable_baselines3.common.distributions import Distribution
from tsGaussian import torch_tsgaussian, utils
from torch import nn, Tensor
from stable_baselines3.sac.policies import Actor as SACActor, MlpPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models import PointNet

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
        print('mu: ', self.mu.size())
        print('sigma: ', self.sigma.size())
        log_prob = self.distribution.log_probs(actions, self.mu, self.sigma) # Parameter for log_probs? R_mu, R_x?
        return log_prob

    def entropy(self) -> Tensor:
        raise NotImplementedError('Not needed.')

    def sample(self) -> Tensor:
        """Need to be implemented"""
        print('self.sigma size: ', self.sigma.size())
        s = self.distribution.rsample(self.mu, self.sigma)
        return s

    """What is this mode function"""
    def mode(self) -> Tensor:
        return self.mu


    """Need to be clear with these functions parameters"""
    def actions_from_params(self, mu: Tensor, sigma: Tensor,
                            deterministic: bool = False) -> Tensor:
        self.proba_distribution(mu, sigma)
        actions = self.get_actions(deterministic = deterministic)
        print('actions size: ', actions.size())
        print('actions type: ', type(actions))
        return self.get_actions(deterministic = deterministic)

    def log_prob_from_params(self, mu, sigma):
        actions = self.actions_from_params(mu, sigma)
        print('actions: ', actions.size())
        log_prob = self.log_prob(actions)
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
        self.vec16 = nn.Linear(last_layer_dim, 12)
        self.action_dist = None

    def get_std(self) -> Tensor:
        raise NotImplementedError('Not needed.')

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotIMplementedError('Not needed.')

    def get_action_dist_params(self, obs: Tensor) -> Tuple[
        Tensor, Tensor, Dict[str, Tensor]]:
        if self.action_dist is None:
            self.action_dist = TangentSpaceGaussian(self.device)
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        vec12 = self.vec16(latent_pi)
        mu, sigma = utils.vec12_to_mu_sigma(vec12)
        print(mu.size())
        print(sigma.size())
        return mu, sigma, {}

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        mu, sigma, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mu, sigma,
                                                    deterministic = deterministic,
                                                    **kwargs)

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

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self._model = PointNet(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._model(observations)
