import torch


def debug_transitions(self, truncated):
    """ """
    critic, done, action_probs, reward, action = self[
        "critic", "env/done", "action_probs", "env/reward", "action"
    ]
    timestep = self["env/timestep"]
    assert not done[
        0
    ].max()  # dones is must be always false in the first timestep of the transition.
    # if not it means we have a transition (step final) => (step initial)

    # timesteps must always follow each other.
    assert (timestep[0] == timestep[1] - 1).all()

    assert (
        truncated[not done].sum().item() == 0
    )  # when done is false, truncated is always false (same for terminated)

    if done[truncated].numel() > 0:
        assert torch.amin(
            done[truncated]
        )  # when truncated is true, done is always true (same for terminated)
    assert reward[1].sum() == len(
        reward[1]
    ), "in cartpole, rewards are always 1"  # only 1 rewards # not general enough
