"""
Tips:

RL, data is collected through interactions with the environment by the agent itself, 
this could lead to vicious circle, so RL may vary from one run to another. 
always do several runs to have quantitative results.

RL are generally dependent on finding appropriate hyperparameters.
A best practice when you apply RL to a new problem is to do automatic hyperparameter optimization. 

When applying RL to a custom problem, you should always normalize the input to the agent,
and look at common preprocessing done on other environments

This reward engineering, necessitates several iterations,  
Deep Mimic combines imitation learning and reinforcement learning to do acrobatic moves.

RL is the instability of training. You can observe during training a huge drop in performance. 
This behavior is particularly present in DDPG, that's why its extension TD3 tries to tackle that issue. 
Other method, like TRPO or PPO make use of a trust region to minimize that problem by avoiding too large update.

Because most algorithms use exploration noise during training, 
you need a separate test environment to evaluate the performance of your agent at a given time. 
It is recommended to periodically evaluate your agent for n test episodes (n is usually between 5 and 20) 
and average the reward per episode to have a good estimate.
"""