Aug 2: implemented a random-sample one-step tabular Q-planning algorithm based on the tensorflow-reinforce project, but found it does not converge at all. One reason is by random sampling and just roll out one step, the model has little of getting correct reward.

At each iteration, the planner does not use the new state from taking the action. Instead, it just select a new state at random. Is this an efficient way of exploring the state space?

Roboschool install notes:
yum install ffmpeq
