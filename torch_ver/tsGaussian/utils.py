import torch
from torch import nn
from tsGaussian.gram_schmidt import gram_schmidt

def vec12_to_mu_sigma(output):
    bs = output.shape[0]
    sigma = output[:, :3]
    print('mu before: ', output[:, 3:].reshape(bs, 3, 3))
    mu = gram_schmidt(output[:, 3:].reshape(bs, 3, 3)) # Some Orthogonal projection process
    print('mu after: ', mu)
    softPlus = nn.Softplus()
    print('sigma after: ', sigma)
    return mu, softPlus(sigma)
