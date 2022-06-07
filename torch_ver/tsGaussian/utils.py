import torch
import torch.nn.functional as F
from torch import nn
from tsGaussian.gram_schmidt import gram_schmidt

def vec7_to_mu_sigma(output):
    bs = output.shape[0]
    sigma = output[:, :3]
    # print('mu before: ', output[:, 3:].reshape(bs, 3, 3))
    # mu = gram_schmidt(output[:, 3:].reshape(bs, 3, 3)) # Some Orthogonal projection process
    # print('mu after: ', mu)
    # mu = torch.eye(3)
    # mu = mu.reshape((1, 3, 3))
    # mu = mu.repeat(bs, 1, 1)
    mu = F.normalize(output[:, 3:])
    softPlus = nn.Softplus()
    print('sigma before: ', sigma[0])
    print('sigma after: ', softPlus(sigma)[0])
    sigma = softPlus(sigma)
    # sigma = 10 * torch.tanh(sigma)
    sigma = torch.exp(sigma)
    return mu, sigma
