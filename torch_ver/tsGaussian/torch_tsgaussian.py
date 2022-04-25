import torch
import numpy as np
from liegroups.torch import SO3

class TangentSpaceGaussian(object):
    """ Finish the batch version of tangent space Gaussian """
    def __init__(self, device: torch.device):
        return

    def vec_to_skew(self, omega: torch.Tensor) -> torch.Tensor:
        """ Convert vector to skew symmetric matrices """
        (*batch_axes, dim) = omega.shape
        assert dim == 3

        omega_hat = torch.zeros((*batch_axes, dim, dim))
        omega_hat[..., 0, 1] = -omega[..., 2]
        omega_hat[..., 1, 0] = omega[..., 2]
        omega_hat[..., 0, 2] = omega[..., 1]
        omega_hat[..., 2, 0] = -omega[..., 1]
        omega_hat[..., 1, 2] = -omega[..., 0]
        omega_hat[..., 2, 1] = omega[..., 0]

        assert omega_hat.shape == (*batch_axes, dim, dim)
        return omega_hat

    def skew_to_vec(self, omiga_hat: torch.Tensor) -> torch.Tensor:
        (*batch_axes, dim, dim) = omega_hat.shape
        assert dim == 3

        omega = torch.zeros((*batch_axes, dim))
        omega[..., 0] = omega_hat[..., 2, 1]
        omega[..., 1] = omega_hat[..., 0, 2]
        omega[..., 2] = omega_hat[..., 1, 0]

        return omega

    def rsample(self, R_mu, sigma):
        """ Sample a rotation using tangent space Gaussian
            Return a skew symmetric matrix.
        """
        omega = torch.normal(torch.zeros(3), sigma)
        R_x = torch.matmul(R_mu, SO3.exp(omega).as_matrix())
        return self.vec_to_skew(R_x)

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
        print(torch.bmm(torch.transpose(R_1, 1, 2), R_2).size())
        return SO3.log(SO3(torch.bmm(torch.transpose(R_1, 1, 2), R_2)))

    def log_probs(self, R_x, R_mu, sigma):
        """ Log probability of a given R_x with mean R_mu
            Return a probability
        """
        log_term = self.log_map(R_x, R_mu)
        # print('log size: ', log_term.size())
        batch_size = R_x.shape[0]
        # print(batch_size)
        sigma_mat = torch.diag_embed(sigma).repeat(batch_size, 1, 1)
        # print(log_term.reshape((batch_size, 1, log_term.shape[1])).size())
        # print(torch.linalg.inv(sigma_mat).size())
        # print(log_term.size())
        log_prob = torch.bmm(torch.bmm(log_term.reshape((batch_size, 1, log_term.shape[1])), torch.linalg.inv(sigma_mat)), \
                    log_term.reshape(batch_size, log_term.shape[1], 1)) - torch.log(self.normal_term(sigma))
        return log_prob.reshape((batch_size,))
