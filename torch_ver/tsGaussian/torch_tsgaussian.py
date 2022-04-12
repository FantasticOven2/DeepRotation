import torch
import numpy as np
from liegroups.torch import SO3

"""Need to make a batch version of this class"""
class TangentSpaceGaussian(object):
    def __init__(self, device: torch.device):
        return
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
