import torch
import numpy as np
# from liegroups.torch import SO3
# from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms.so3 import so3_exp_map, so3_log_map
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

"""Test liegroup exp: correct"""
"""Test liegroup log: correct (should be correct based on relationship between exp and log)"""
"""Test scipy transform: #TODO"""
class TangentSpaceGaussian(object):
    """ Finish the batch version of tangent space Gaussian """
    def __init__(self, device: torch.device):
        return
    #
    # def vec_to_skew(self, omega: torch.Tensor) -> torch.Tensor:
    #     """ Convert vector to skew symmetric matrices """
    #     (*batch_axes, dim) = omega.shape
    #     assert dim == 3
    #
    #     omega_hat = torch.zeros((*batch_axes, dim, dim))
    #     omega_hat[..., 0, 1] = -omega[..., 2]
    #     omega_hat[..., 1, 0] = omega[..., 2]
    #     omega_hat[..., 0, 2] = omega[..., 1]
    #     omega_hat[..., 2, 0] = -omega[..., 1]
    #     omega_hat[..., 1, 2] = -omega[..., 0]
    #     omega_hat[..., 2, 1] = omega[..., 0]
    #
    #     assert omega_hat.shape == (*batch_axes, dim, dim)
    #     return omega_hat

    # def skew_to_vec(self, omiga_hat: torch.Tensor) -> torch.Tensor:
    #     (*batch_axes, dim, dim) = omega_hat.shape
    #     assert dim == 3
    #
    #     omega = torch.zeros((*batch_axes, dim))
    #     omega[..., 0] = omega_hat[..., 2, 1]
    #     omega[..., 1] = omega_hat[..., 0, 2]
    #     omega[..., 2] = omega_hat[..., 1, 0]
    #
    #     return omega

    # def mat_to_quat(self, mat: torch.Tensor) -> torch.Tensor:


    def rsample(self, R_mu, sigma):
        """ Sample a rotation using tangent space Gaussian
            Return a skew symmetric matrix.
        """
        print('R_mu: ', R_mu[0])
        dev = sigma.get_device()
        if dev == -1:
            dev = 'cpu'
        omega = torch.normal(torch.zeros(3, device=dev), sigma)
        R_mu = so3_exp_map(R_mu).to(dev)
        R_x = torch.matmul(R_mu, so3_exp_map(omega))
        R_quat = matrix_to_quaternion(R_x)
        return R_quat, R_x

    def normal_term(self, sigma):
        """ Compute normalization term in the pdf of tangent space Gaussian
            Return a scalar
        """
        sigma_mat = torch.diag_embed(sigma)
        return torch.sqrt((2 * np.pi) ** 3 * torch.det(sigma_mat))

    def log_map(self, R_1, R_2):
        """ Log map term in pdf of tangent space Gaussian
            Return a 3d vector.
        """
        return so3_log_map(torch.bmm(torch.transpose(R_1, 1, 2), R_2), eps = 0.0001)

    def log_probs(self, R_x, R_mu, sigma):
        """ Log probability of a given R_x with mean R_mu
            Return a probability
        """
        dev = R_x.get_device()
        if dev == -1:
            dev = 'cpu'
        R_mu = so3_exp_map(R_mu).to(dev)
        log_term = self.log_map(R_x, R_mu)
        batch_size = R_x.shape[0]
        sigma_mat = torch.diag_embed(sigma)
        log_prob = -(torch.bmm(torch.bmm(log_term.reshape((batch_size, 1, 3)), torch.linalg.inv(sigma_mat)), \
                    log_term.reshape(batch_size, 3, 1)).reshape((batch_size,))) / 2 - torch.log(self.normal_term(sigma))
        return log_prob

    def entropy(self, mu, sigma):
        sigma_mat = torch.diag_embed(sigma)
        return 3 * (1 + torch.log(2 * np.pi)) / 2 + torch.log(torch.det(sigma_mat)) / 2