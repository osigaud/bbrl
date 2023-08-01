from pathlib import Path
import re
import gym
import torch


def load_agent(path: Path, prefix: str):
    """Loads an agent in the given folder with the highest reward

    Agent files should have the pattern '{prefix}_REWARD.agt'
    """
    argmax_r, max_r = None, float("-inf")
    for p in path.glob(f"{prefix}*.agt"):
        m = re.match(rf".*/{prefix}(-?\d+\.\d+)\.agt", str(p))
        r = float(m.group(1))
        if r > max_r:
            max_r = r
            argmax_r = p

    if argmax_r:
        print(f"Loading {argmax_r}")
        return torch.load(argmax_r)
    return None


def play(env: gym.Env, agent: torch.nn.Module):
    """Render the agent"""
    if agent is None:
        print("No agent")
        return

    sum_reward = 0.0

    try:
        print(agent)
        with torch.no_grad():
            obs = env.reset()
            env.render()
            done = False
            while not done:
                obs = torch.Tensor(obs)
                action = agent.predict_action(obs, False)
                obs, reward, done, info = env.step(action.numpy())
                sum_reward += reward
                env.render()
    finally:
        env.close()

    return reward
