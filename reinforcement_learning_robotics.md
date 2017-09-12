---
title: RL and Robotics
---

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

### Gaussian process dynamic programming.pdf
Model the value function and state value function as a GP, and use Q-Learning like methods. Also see references in this paper for model-based methods that use GP to approximate the model.

This work also use GP as a policy approximation.

### Gaussian processes in reinforcement learning
Use GP as approximation of the model. Learn model from data. Model-based policy iteration.

### Probabilistic Inference for Fast Learning in Control
Not like above paper, this work directly optimize policy. 
