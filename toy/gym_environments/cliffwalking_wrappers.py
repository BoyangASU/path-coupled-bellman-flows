import gymnasium as gym


class CliffWalkingTerminatesOnFallWrapper(gym.Wrapper):
    """
    A wrapper for CliffWalking-v0 that terminates the episode if the agent
    falls off the cliff. It can also modify the penalty for falling.
    In the original environment, falling off the cliff incurs a large negative
    reward but does not end the episode.
    """

    def __init__(self, env, fall_penalty: float = -1.0):
        """
        Initialize the wrapper.

        Args:
            env: The CliffWalking-v0 environment.
            fall_penalty (float): The reward to give when the agent falls.
                                  Defaults to -1.0.
        """
        super().__init__(env)
        self.fall_penalty = fall_penalty

    def step(self, action):
        """
        Overrides the step method to check for the cliff fall condition
        and modify the reward.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        # In CliffWalking-v0, a reward of -100 is given for falling off the cliff.
        if reward == -100.0:
            terminated = True
            reward = self.fall_penalty
        return observation, reward, terminated, truncated, info


class CliffWalkingNeverTerminatesWrapper(gym.Wrapper):
    """
    A wrapper for CliffWalking-v0 that creates a continuing task.
    When the goal is reached, the episode doesn't terminate. Instead, the
    agent is reset to the starting position, and the episode continues.
    """

    def __init__(self, env):
        super().__init__(env)
        # This wrapper makes the task continuing.
        # Setting max_episode_steps to None can prevent auto-reset from
        # wrappers like TimeLimit, though CliffWalking-v0 has no default limit.
        if self.env.spec:
            self.env.spec.max_episode_steps = None

    def step(self, action):
        """
        Overrides the step method to handle the terminal state. When the goal
        is reached, it resets the environment and continues the episode.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            # The original environment terminated (goal reached).
            # We reset the environment to continue the episode.
            # The observation for the next state is the starting state.
            observation, info_reset = self.env.reset()
            info.update(info_reset)
            # The episode itself does not terminate.
            terminated = False

        return observation, reward, terminated, truncated, info 


if __name__ == "__main__":
    print("Testing CliffWalking wrappers...")
    
    # Test 1: CliffWalkingTerminatesOnFallWrapper
    print("\n--- Testing CliffWalkingTerminatesOnFallWrapper ---")
    env_terminates = CliffWalkingTerminatesOnFallWrapper(gym.make('CliffWalking-v1'), fall_penalty=-1.0)
    obs, info = env_terminates.reset()
    assert obs == 36, "Initial state should be 36"
    print(f"Initial state: {obs}")

    # Action 1 (RIGHT) from start state (36) leads to falling off the cliff.
    next_obs, reward, terminated, truncated, info = env_terminates.step(1)
    
    print(f"Action: RIGHT -> Next State: {next_obs}, Reward: {reward}, Terminated: {terminated}")
    assert reward == -1.0, "Reward for falling should now be -1.0"
    assert terminated is True, "Episode should terminate upon falling into the cliff"
    print("✅ CliffWalkingTerminatesOnFallWrapper test passed!")
    env_terminates.close()

    # Test 2: CliffWalkingNeverTerminatesWrapper
    print("\n--- Testing CliffWalkingNeverTerminatesWrapper ---")
    env_continuous = CliffWalkingNeverTerminatesWrapper(gym.make('CliffWalking-v0'))
    obs, info = env_continuous.reset()
    assert obs == 36, "Initial state should be 36"
    print(f"Initial state: {obs}")
    
    # A safe path to the goal: UP, 11 * RIGHT, DOWN
    actions_to_goal = [0] + [1] * 11 + [2]
    
    print(f"Executing safe path to goal...")
    for i, action in enumerate(actions_to_goal):
        obs, reward, terminated, truncated, info = env_continuous.step(action)
        if i == len(actions_to_goal) - 1:
            print("Final step to reach goal:")
            print(f"  - Action: {action}, Next State: {obs}, Reward: {reward}, Terminated: {terminated}")

    assert obs == 36, "After reaching goal, agent should be reset to start state (36)"
    assert terminated is False, "Episode should not terminate upon reaching the goal"
    
    # Verify we can continue the episode
    next_obs, reward, terminated, truncated, info = env_continuous.step(0) # Step UP from start
    print("Taking one more step after reaching goal...")
    print(f"Action: UP -> Next State: {next_obs}, Reward: {reward}, Terminated: {terminated}")
    assert next_obs == 24, "Agent should have moved from start to state 24"
    assert terminated is False, "Episode should continue after reset"

    print("✅ CliffWalkingNeverTerminatesWrapper test passed!")
    env_continuous.close()
    
    print("\nFor full Monte Carlo analysis, please run 'test_all_mc_sampling.py'") 