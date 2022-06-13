"""
Wahba env inspired by:
Peretroukhin, Valentin, et al. "A smooth representation of belief over so (3)
for deep rotation learning with uncertainty." RSS 2020
"""

import gym
import numpy as np
import torch
from gym import spaces
from liegroups.torch import SO3 as SO3_torch
from tsGaussian.utils import *
from tsGaussian.gram_schmidt import gram_schmidt

N_MATCHES_PER_SAMPLE = 100


def rotmat_to_quat(mat, ordering='xyzw'):
    """Convert a rotation matrix to a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if mat.dim() < 3:
        R = mat.unsqueeze(dim=0)
    else:
        R = mat

    assert (R.shape[1] == R.shape[2])
    assert (R.shape[1] == 3)

    # Row first operation
    R = R.transpose(1, 2)
    q = R.new_empty((R.shape[0], 4))

    cond1_mask = R[:, 2, 2] < 0.
    cond1a_mask = R[:, 0, 0] > R[:, 1, 1]
    cond1b_mask = R[:, 0, 0] < -R[:, 1, 1]

    if ordering == 'xyzw':
        v_ind = torch.arange(0, 3)
        w_ind = 3
    else:
        v_ind = torch.arange(1, 4)
        w_ind = 0

    mask = cond1_mask & cond1a_mask
    if mask.any():
        t = 1 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]
        q[mask, w_ind] = R[mask, 1, 2] - R[mask, 2, 1]
        q[mask, v_ind[0]] = t
        q[mask, v_ind[1]] = R[mask, 0, 1] + R[mask, 1, 0]
        q[mask, v_ind[2]] = R[mask, 2, 0] + R[mask, 0, 2]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask & cond1a_mask.logical_not()
    if mask.any():
        t = 1 - R[mask, 0, 0] + R[mask, 1, 1] - R[mask, 2, 2]
        q[mask, w_ind] = R[mask, 2, 0] - R[mask, 0, 2]
        q[mask, v_ind[0]] = R[mask, 0, 1] + R[mask, 1, 0]
        q[mask, v_ind[1]] = t
        q[mask, v_ind[2]] = R[mask, 1, 2] + R[mask, 2, 1]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask.logical_not() & cond1b_mask
    if mask.any():
        t = 1 - R[mask, 0, 0] - R[mask, 1, 1] + R[mask, 2, 2]
        q[mask, w_ind] = R[mask, 0, 1] - R[mask, 1, 0]
        q[mask, v_ind[0]] = R[mask, 2, 0] + R[mask, 0, 2]
        q[mask, v_ind[1]] = R[mask, 1, 2] + R[mask, 2, 1]
        q[mask, v_ind[2]] = t
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask.logical_not() & cond1b_mask.logical_not()
    if mask.any():
        t = 1 + R[mask, 0, 0] + R[mask, 1, 1] + R[mask, 2, 2]
        q[mask, w_ind] = t
        q[mask, v_ind[0]] = R[mask, 1, 2] - R[mask, 2, 1]
        q[mask, v_ind[1]] = R[mask, 2, 0] - R[mask, 0, 2]
        q[mask, v_ind[2]] = R[mask, 0, 1] - R[mask, 1, 0]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    return q.squeeze()


def quat_norm_diff(q_a, q_b):
    assert (q_a.shape == q_b.shape)
    assert (q_a.shape[-1] == 4)
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    print('difference: ', (q_a - q_b).norm(dim=1))
    print('summation: ', (q_a + q_b).norm(dim=1))
    return torch.min((q_a - q_b).norm(dim=1), (q_a + q_b).norm(dim=1)).squeeze()


def quat_chordal_squared_loss(q, q_target, reduce=True):
    assert (q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    print(d)
    print('d shape: ', d.shape)
    losses = 2 * d * d * (4. - d * d)
    # losses = d / 2
    print('losses: ', losses)
    loss = losses.mean() if reduce else losses
    # print(loss)
    return loss

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    # print(type(m1))
    # print(type(m2))
    # m1 = torch.from_numpy(m1).float()
    if m1.dim() < 3:
        m1 = m1.unsqueeze(0)
    # print(m1)
    # print(m1[0,0,0])
    # print('m2 shape: ', m2.shape)
    m = torch.bmm(m1, m2.float().transpose(1, 2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch))*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    # print(theta.shape)
    
    return theta

def fro_loss(R1, R2):
    return torch.norm(R1 - R2, 'fro').mean()

class Wahba(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Wahba, self).__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 3, N_MATCHES_PER_SAMPLE),
            dtype=np.float32)
        self._obs, self._action = None, None

    def _gen_sim_data_fast(self, N_rotations, N_matches_per_rotation, sigma,
                           max_rotation_angle=None, dtype=torch.double):
        axis = torch.randn(N_rotations, 3, dtype=dtype)
        axis = axis / axis.norm(dim=1, keepdim=True)
        if max_rotation_angle:
            max_angle = max_rotation_angle * np.pi / 180.
        else:
            max_angle = np.pi
        angle = max_angle * torch.rand(N_rotations, 1)
        C = SO3_torch.exp(angle * axis).as_matrix()
        if N_rotations == 1:
            C = C.unsqueeze(dim=0)
        x_1 = torch.randn(N_rotations, 3, N_matches_per_rotation, dtype=dtype)
        x_1 = x_1 / x_1.norm(dim=1, keepdim=True)
        ### NEED TO CHANGE
        noise = sigma * torch.randn_like(x_1)
        x_2 = C.bmm(x_1) + noise
        return C, x_1, x_2

    def step(self, action):

        # action /= np.linalg.norm(action, axis=0)
        # w, x, y, z = action

        # loss = quat_chordal_squared_loss(
        #     torch.tensor([x, y, z, w], dtype=self._q_target.dtype),
        #     self._q_target)
        action = torch.Tensor(action)
        if action.dim() < 2:
            action = action.unsqueeze(0)
        bs = action.shape[0]
        # action = gram_schmidt(action.reshape((bs, 3, 3)))
        rot6d, axis_angle = action[:, 3 :], action[:, : 3]
        R_mu = compute_rotation_matrix_from_ortho6d(rot6d)
        R_x = compute_rotation_matrix_from_Rodriguez(axis_angle)
        action = torch.bmm(R_mu, R_x)
        # loss = fro_loss(action, self._q_target)
        # print('loss: ', loss)
        loss = compute_geodesic_distance_from_two_matrices(action, self._q_target).mean()
        rew = -loss.item()
        # print('rew: ', rew)
        return self._obs, rew, True, {}

    def reset(self):
        C_train, x_1_train, x_2_train = self._gen_sim_data_fast(
            1, N_MATCHES_PER_SAMPLE, 1e-2, max_rotation_angle=180)
        # self._q_target = rotmat_to_quat(C_train, ordering='xyzw')
        # print(type(C_train))
        self._q_target = C_train
        # self._q_target = rotmat_to_quat(torch.eye(3).reshape(1,3,3), ordering='xyzw')
        self._obs = np.concatenate([x_1_train, x_2_train])
        # self._obs = np.concatenate([x_1_train, x_1_train])
        # self._obs = np.concatenate([torch.ones(1, 3, 100), torch.ones(1, 3, 100)])
        return self._obs

    def render(self, mode='human', close=False):
        pass
