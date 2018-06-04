Aug 2: implemented a random-sample one-step tabular Q-planning algorithm based on the tensorflow-reinforce project, but found it does not converge at all. One reason is by random sampling and just roll out one step, the model has little of getting correct reward.

At each iteration, the planner does not use the new state from taking the action. Instead, it just select a new state at random. Is this an efficient way of exploring the state space?

Roboschool install notes:
yum install ffmpeq

Gradient boosting explicitly calculate the residual signal for each weak model, and improve the model by fitting the residual with an additional model. The concept of residual signal seems also related to residual network, where a residual signal path is defined as a shortcut to the standard convolution operation. However, in residual network, there is no explicitly iteration over the residual signals. That is, the base convolution and the residual signal is learned jointly. So, can we improve any residual based deep net and learn it in an incremental way just like gradient boosting, so the model is guaranteed to improve over previous base model?
