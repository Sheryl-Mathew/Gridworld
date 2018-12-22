# Gridworld

## World

You are given a Gridworld environment that is defined as follows:

State space: GridWorld has 10x10 = 100 distinct states. The start state is the top left cell. The gray cells are walls and cannot be moved to.

Actions: The agent can choose from up to 4 actions (left, right, up, down) to move around.

Environment Dynamics: GridWorld is deterministic, leading to the same new state given
each state and action

Rewards: The agent receives +1 reward when it is in the center square (the one that shows R 1.0), and -1 reward in a few states (R -1.0 is shown for these). The state with +1.0 reward
is the goal state and resets the agent back to start. In other words, this is a deterministic, finite Markov Decision Process (MDP). Assume the
discount factor beta=0.9. Assume a small fixed learning rate aplha=0.0

## Implementation

Implement the Q-learning algorithm to learn the Q values for each state-action pair.

Experiment with different explore/exploit policies:

1) Ephsilon-greedy. Try Ephsilon values 0.1, 0.2, and 0.3.

2) Boltzman exploration. Start with a large temperature value T and follow a fixed scheduling rate. Give these details in your report.

How many iterations did it take to reach convergence with different exploration policies?

Please show the converged Q values for each state-action pair.