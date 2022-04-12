import torch
import numpy as np
from liegroups.torch import SO3

"""Need to make a batch version of this class"""
class TangentSpaceGaussian(object):
    def __init__(self, device: torch.device):
        return

    def skew(omega: torch.Tensor) -> torch.Tensor:
        (*batch_axes, dim) = omega.shape
        assert dim == 3

        omega_hat = torch.zeros((*batch_axes, 3, 3))
        omega_hat[..., 0, 1] = -omega[..., 2]
        omega_hat[..., 1, 0] = omega[..., 2]
        omega_hat[..., 0, 2] = omega[..., 1]
        omega_hat[..., 2, 0] = -omega[..., 1]
        omega_hat[..., 1, 2] = -omega[..., 0]
        omega_hat[..., 2, 1] = omega[..., 0]

        assert omega_hat.shape == (*batch_axes, dim, dim)
        return omega_hat

    """Sampling from the Gaussian N(0, sigma) then project it back on to SO(3)"""
    """Issue: Code runs, need to check correctness"""
    def rsample(self, R_mu, sigma):
        omiga = torch.normal(torch.zeros(3), sigma)
        print('omiga: ', omiga.size())
        omiga_0, omiga_1, omiga_2 = omiga[0][0], omiga[1][1], omiga[2][2]
        omiga_hat = torch.tensor([[0, -omiga_2, omiga_1],
                                    [omiga_2, 0, -omiga_0],
                                    [-omiga_1, omiga_0, 0]])
        print(SO3.exp(omiga_hat).as_matrix().size())
        R_x = R_mu @ SO3.exp(omiga_hat).as_matrix().reshape((1,3,3))
        return R_x

    """Issue: Code runs, need to check corretness"""
    def normal_term(self, sigma):
        return torch.sqrt((2 * np.pi) ** 3 * torch.det(sigma))

    """Issue: Need clarification"""
    def log_map(self, R_1, R_2):
        return SO3.log(SO3(R_1.T @ R_2))

    """Issue: Code runs, need to check correctness"""
    def log_probs(self, R_x, R_mu, sigma):
        log_term = self.log_map(R_x, R_mu)
        return log_term.T @ torch.linalg.inv(sigma) @ log_term - \
                torch.log(self.normal_term(sigma))
