# Dealing with time limits

A lot of RL problems are episodic: the agent must achieve a task and when this is done, the episode stops. To deal with the case where the agent may not succeed even given an infinite time, and to accelerate learning, it is standard for these episodic problems to come with a time limit: if the agent did not succeed after a number of time steps, the episode stops.

As explained in [this paper](http://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf), this situation is a source of instability in RL, as it creates some non-stationarity in the underlying MDP. The point is that, in the same state, the agent may either continue and receive some later reward, or get stopped by the time limit and receive nothing.

The proper way to deal with time limits consists in still propagating values in the critic from the next state to the current state (a Bellman backup) over the last transition when the episode is stopped by a time limit, by contrast with the case where the episode stops because the task is done, in which case the value of the next state should be ignored.

In former [OpenAI gym environments](https://www.gymlibrary.dev/index.html), properly dealing with time limits was a little intricate. With the more recent [gymnasium](https://gymnasium.farama.org/index.html) library, the situation is simpler. The environment outputs three variables related to the end of an episode:
- `terminated` is True if the episode stops due to a terminal success or failure from the behavior of the agent, False otherwise;
- `truncated` is True if the episode stops because the time limit is elapsed, False otherwise;
- `done` is True if either terminated or truncated is true, False otherwise.

So the rules to apply when an episode stops is simple: the values from the previous step should not be propagated if `terminated` is True, and it should be propagated in any other case.

To implement the above, rather than using complicated "if... else... " rules, we multiply the value of the next state by `~terminated`, making profit of the True=1, False=0 equivalence in python: if the `terminated` boolean is True, its value is 1, thus `~terminated` is 0 and the value from the next state is cancelled. If it is False, `~terminated` is 1, thus the value is not cancelled.
