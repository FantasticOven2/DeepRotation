import gym
from gym import spaces

import jax.numpy as jnp
from jax import random

N_MATCHES_PER_SAMPLE = 100

"""Details subject to change"""
def rotmat_to_quat(mat, ordering = 'xyzw'):
    """Convert a rotation matrix to a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'
    """
    if mat.dim() < 3:
        R = mat.unsqueeze(dim = 0)
    else:
        R = mat

    assert(R.shape[1] == R.shape[2])
    assert(R.shape[1] == 3)

    # Row first operation
    R = R.transpose(1, 2)
    q = R.new_empty((R.shape[0], 4))

    cond1_mask = R[:, 2, 2] < 0
    cond1a_mask = R[:, 0, 0] > R[:, 1, 1]
    cond1b_mask = R[:, 0, 0] < -R[:, 1, 1]

    if ordering == 'xyzw':
        v_ind = torch.arange(0, 3)
        w_ind = 3
    else:
        v_ind = torch.arange(1, 4)
        w_ind = 0

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
    return jnp.min((q_a - q_b).norm(dim = 1), (q_a + q_b).norm(dim = 1)).squeeze()

def quat_chordal_squared_loss(q, q_target, reduce = True):
    assert (q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses = 2 * d * d * (4. - d * d)
    loss = losses.mean() if reduce else losses
    return loss

class Wahba(gym.Env):
    """Custom Wahba Environment in jax"""

    # To jax
    def __init__(self):
        super(Wahba, self).__init__()
        self.action_space = spaces.Box(
            low = -1.0, high = 1.0, shape = (4,), dtype = jnp.float32)
        self.observation_space = spaces.Box(
            low = 1.0, high = 1.0, shape = (2,3, N_MATCHES_PER_SAMPLE),
            dtype = jnp.float32)
        self.obs, self._action = None, None

    def randm_action(self):
        return self.action_space.sample()


    """Details subject to change"""
    def _gen_sim_data_fast(self, N_rotations, N_matches_per_rotation, sigma,
                            max_rotation_angle = None, dtype = torch.double):
        key = random.PRNGKey(0)
        axis = random.uniform(key, (N_rotations, 3), dtype = dtype)
        axis = axis / axis.norm(dim = 1, keepdim = True)
        if max_rotation_angle:
            max_angle = max_rotation_angle * jnp.pi / 180.
        else:
            max_angle = jnp.pi
        angle = max_angle * random.uniform(key, (N_rotations, 1))
        C = SO3_torch.exp(angle * axis).as_matrix() ### Need to be changed later
        if N_rotations == 1:
            C = C.unsqueeze(dim = 0)
        x_1 = random.uniform(key, (N_rotations, 3, N_matches_per_rotation), dtype = dtype)
        x_1 = x_1 / x_1.norm(dim = 1, keepdim = True)
        noise = sigma * random.uniform(key, x_1.shape)
        x_2 = C.bmm(x_1) + noise
        return C, x_1, x_2

    # To jax
    def step(self, action):
        action /= jnp.linalg.norm(action, axis = 0)
        w, x, y, z = action
        loss = quat_chordal_squared_loss(
                jnp.array([x, y, z, w], dtype = self._q_target.dtype),
                self._q_target)
        print(type(rew))
        rew = -loss.item()
        return self._obs, rew, True, {}

    # To jax
    def reset(self):
        C_train, x_1_train, x_2_train = self._gen_sim_data_fast(
            1, N_MATCHES_PER_SAMPLE, 1e-2, max_rotation_angle = 180)
        self._q_target = rotmat_to_quat(C_train, ordering = 'xyzw')
        self._obs =  jnp.concatenate([x_1_train, x_2_train])
        return self._obs


def main(argv):
    env = Wahba()
    env.step(env.random_action())
    env.reset()
