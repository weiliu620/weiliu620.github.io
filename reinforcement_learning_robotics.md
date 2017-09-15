<!-- ---
title: RL and Robotics
--- -->

### Deep Learning for Detecting Robotic Grasps

### Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection

### On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models

### Compressed Network Search Finds Complex Neural Controllers with a Million Weights

### Value Iteration Network
If action moves agent locally, then can be computed with convolution.

### feedback Network

Iterative representations based on feedback from previous iteration's output. Benefits: early prediction, hierarchical output (taxonomy), provide basis for curriculum learning. [what is it]

### Cognitive Mapping and Planning for Visual Navigation

J. Malik's lab. Mapping is driven by the need of planner. Spatial memory, can plan with incomplete observations of the world. Agent has a belief map of the world. [heavy in mapping, skip it for now]

### DeepMPC: Learning Deep Latent Features for Model Predictive Control

Ian Lenz's paper.

### Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control

Sent to Tom but haven't read it yet.

### Gaussian Process Model Based Predictive Control

### Predictive control with Gaussian process models

### Deep Recurrent Q-Learning for Partially Observable MDPs
Peter Stone's work


### Deep Reinforcement Learning with Double Q-learning
Original EQN contains a maximization step over estimated values, and this max step is an biased towards higher values. The paper shows DQN has overestimates even when network is deep and environment is deterministic.

### gaussian processes for data-efficient learning in robotics and control
Use Gaussian process to model the transition probability, derive a closed form expression for expected reward/cost for policy evaluation, and analytic policy gradients for policy improvement.

But since we have Tensorflow and other libraries to calculate the gradient automatically, can we still define GP for the model, analytically derive the expected reward (i.e. value function) and use Tf to calculate and update its gradient?

### Approximate Dynamic Programming with Gaussian Processes.pdf (1998)
Assuming transition model is know, use GP to approximate both value function V and action-value function Q. With such approximation, Both GPs learn from discrete state and action pairs, but can be evaluated at continuous action value.

### Gaussian process dynamic programming
Model the value function and state value function as a GP, and use Q-Learning like methods.

This work also use GP as a policy approximation. It is an extension of the above paper in 1998, in that this work also learn the models by using another GP, and online select the state for exploration. The advantage is that we can only explore interesting state and actions along the trajectories, and do not need to sample the regular grid of state and actions like previous paper.  Related to Bayesian optimization and experimental design. In this sense, this GP policy is global, not local.

The first algorithm assumes that model is known, as well as the immediate reward given current state and action. The algorithm does not actually interact with the environment. It just define a set of states sampled from regular grid of state space, and use the assumed transition function to calculate next state and reward. Then use (state, action, new state, reward) to learn Q function approximate by GP.

### Gaussian processes in reinforcement learning
Use GP as approximation of the model. Learn model from data. Model-based policy iteration.

### Probabilistic Inference for Fast Learning in Control
Not like above paper, this work directly optimize policy.

### PILCO/ A Model-Based and Data-Efficient Approach to Policy Search
Proposed to use Gaussian process as an representation of the transition model. Based on GP, calculate the marginal distribution of state, and use that to calculate analytical form of the value function. Then, derive an analytical form for the gradient of the value function with regard to the the policy function's parameters.

The cost/value function is defined such that the expectation with respect to the state distribution is in closed form. This need cost function be in a special form. In this paper it's exponential function of (x - x_target). Such different is different from OpenAI gym, which gives the reward when each action is taken. the environment in gym is like a blackbox, and algorithm designer cannot change definition of reward. On the other hand, the method used in this paper make some sense: if in practice we know the optimal state of the problem, we can define the cost function based on the difference of current state and optimal state. This is similar to my previous thinking: redefine the reward function based on the current states and target states.  
