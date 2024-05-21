# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch


def _index(tensor_3d, tensor_2d):
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


def cumulated_reward(reward, done):
    T, B = done.size()
    done = done.detach().clone()

    v_done, index_done = done.float().max(0)
    assert v_done.eq(
        1.0
    ).all(), "[agents.rl.functional.cumulated_reward] Computing cumulated reward over unfinished trajectories"
    arange = torch.arange(T, device=done.device).unsqueeze(-1).repeat(1, B)
    index_done = index_done.unsqueeze(0).repeat(T, 1)

    mask = arange.le(index_done)
    reward = (reward * mask.float()).sum(0)
    return reward.mean().item()


def temporal_difference(critic, reward, must_bootstrap, discount_factor):
    target = discount_factor * critic[1:].detach() * must_bootstrap.float() + reward[1:]
    td = target - critic[:-1]
    to_add = torch.zeros(1, td.size()[1]).to(td.device)
    td = torch.cat([td, to_add], dim=0)
    return td


def doubleqlearning_temporal_difference(
    q, action, q_target, reward, must_bootstrap, discount_factor
):
    action_max = q.max(-1)[1]
    q_target_max = _index(q_target, action_max).detach()[1:]

    mb = must_bootstrap.float()
    target = reward[1:] + discount_factor * q_target_max * mb

    q = _index(q, action)[:-1]
    td = target - q
    to_add = torch.zeros(1, td.size()[1], device=td.device)
    td = torch.cat([td, to_add], dim=0)
    return td


def gae(reward, next_critic: torch.Tensor, must_bootstrap: torch.Tensor, critic: torch.Tensor, discount_factor: float, gae_coef: float):
    """Computes the generalized advantage estimation

    :param reward: The reward matrix for each transition (dimension TxB)
    :param next_critic: The critic value at t+1 (dimension TxB)
    :param must_bootstrap: Must bootstrap flag, true if this was not the last
        state of an episode (dimension TxB)
    :param critic: The critic value at t (dimension TxB)
    :param discount_factor: The discount factor
    :param gae_coef: The generalized advantage lambda
    :return: a TxB matrix containing the advantages
    """
    mb = must_bootstrap.int()
    # delta = reward + discount_factor * next_critic.detach() * mb
    td = reward + discount_factor * next_critic.detach() * mb - critic

    # handling TD0 case
    if gae_coef == 0.0:
        return td

    # Compute GAE
    gae_val = td[-1]
    gaes = [gae_val]
    for t in range(len(td) - 2, -1, -1):
        # print(t, "td", td[t], mb[t])
        gae_val = td[t] + discount_factor * gae_coef * mb[t] * gae_val
        gaes.append(gae_val)
        
    # Returns the matrix of advantages
    gaes.reverse()
    gaes = torch.stack(gaes, dim=0)
    return gaes
