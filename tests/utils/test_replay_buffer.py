import torch
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.workspace import Workspace


def test_replay_buffer_put():
    """Ensures that the replay buffer handles properly transitions"""
    wp = Workspace()
    rb_size = 100
    length = rb_size // 2

    wp.set_full("action", torch.arange(length + 1))
    wp.set_full("action-2", torch.arange(length + 1))
    wp.set_full("env/done", torch.full((length + 1,), 0))

    tr_wp = wp.get_transitions()
    assert tr_wp.batch_size() == length

    rb = ReplayBuffer(100)
    rb.put(tr_wp)

    target = torch.stack((torch.arange(length), torch.arange(1, length + 1)))
    assert (rb.variables["action"][:length].T == target).all()

    data = rb.get_shuffled(50)
    assert (data["action"] == data["action-2"]).all()
    assert not rb.is_full, f" rb real size = {length} / {rb.size()}"
    # assert rb.size() == tr_wp.variables["env/timestep"].size[1]


def test_replay_buffer_overflow():
    """Ensures that the replay buffer handles properly transitions"""
    rb_size = 100
    length = rb_size // 2

    rb = ReplayBuffer(100)

    # --- First insert
    wp = Workspace()
    wp.set_full("action", torch.arange(length + 1))
    wp.set_full("action-2", torch.arange(length + 1))
    wp.set_full("env/done", torch.full((length + 1,), 0))

    tr_wp = wp.get_transitions()
    assert tr_wp.batch_size() == 50
    rb.put(tr_wp)

    # --- Second insert
    wp = Workspace()
    wp.set_full("action", torch.arange(length, length * 2 + 1))
    wp.set_full("action-2", torch.arange(length, length * 2 + 1))
    wp.set_full("env/done", torch.full((length + 1,), 0))

    tr_wp = wp.get_transitions()
    assert tr_wp.batch_size() == 50
    rb.put(tr_wp)

    # --- Third insert
    wp = Workspace()
    wp.set_full("action", torch.arange(length * 2, length * 3 + 1))
    wp.set_full("action-2", torch.arange(length * 2, length * 3 + 1))
    wp.set_full("env/done", torch.full((length + 1,), 0))

    tr_wp = wp.get_transitions()
    assert tr_wp.batch_size() == 50
    rb.put(tr_wp)

    # --- Checks
    assert rb.is_full

    actions = torch.cat(
        (torch.arange(length * 2, length * 3), torch.arange(length, length * 2))
    )
    assert torch.equal(rb.variables["action"][:, 0], actions)
