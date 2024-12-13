# Reinforcement Learning with Snake Game

This project explores **Reinforcement Learning** using two popular algorithms: **Deep Q Network (DQN)** and **Advantage Actor Critic (A2C)**.
The Snake game was chosen as a simple environment to test and compare the performance of these algorithms.
Along the way, I tackled various technical challenges related to training RL models.

![snake_game_video-ezgif com-resize](https://github.com/user-attachments/assets/d1d5cb96-4bb3-4edb-ab4a-32a3be9a9d78)


---

## Project Overview

Reinforcement Learning (RL) differs significantly from traditional supervised learning.
It requires the model to learn by interacting with the environment and discovering which actions yield the highest rewards.
The Snake game, with its straightforward rules and discrete action space, served as an ideal starting point for exploring RL algorithms.
However, this simplicity did not eliminate challenges such as:

- Balancing exploration and exploitation.
- Efficiently utilizing computational resources.
- Managing memory during hyperparameter tuning.

The implementation of DQN and A2C was achieved using **Stable-Baselines3**, which simplified focusing on the algorithms themselves rather than constructing everything from scratch.

---

## Algorithms

### Deep Q Network (DQN)

- Learns a Q-value function that predicts the expected reward for each action in a given state.
- Employs an epsilon-greedy strategy to balance exploration and exploitation.
- Uses a replay buffer to store past experiences, stabilizing the training process.
- Performs well but may struggle with exploration in more complex environments.

### Advantage Actor Critic (A2C)

- Utilizes two separate models: one for the policy (actor) and another for value estimation (critic).
- Actions are sampled from the actor's probability distribution.
- Learns faster than DQN but demands more computational resources and can be harder to stabilize.

---

## Challenges and Solutions

### 1. CPU Bottlenecks

The AWS G4 machine used had a powerful GPU (Tesla T4), but the CPU became a bottleneck when simulating the environment.
The GPU was underutilized, running at less than 30% capacity.

**Solution:**
- Parallelized the environment using Stable-Baselines3 to run simulations on multiple CPU cores. This improved processing speed by approximately 5x.
- However, the supervising process still overloaded a single CPU core, likely due to limitations in Python’s multiprocessing or my implementation.

### 2. GPU Utilization

To better utilize the GPU, I increased the batch size and replay buffer size.
A larger replay buffer allows the model to learn from more past experiences without overloading the CPU.

**Solution:**
- Configured the replay buffer to use up to 50% of available RAM. While this improved stability, replay buffer size didn’t significantly impact training performance.
- Future plan: Explore moving the replay buffer to a NoSQL database for improved memory management.

### 3. Memory Management

When tuning hyperparameters with Optuna, memory leaks occurred due to the replay buffer not being cleared between trials, causing the machine to run out of memory.

**Solution:**
- Manually deleted the model and cleared the GPU cache after each trial, enabling uninterrupted hyperparameter tuning.

---

## Training Results

### DQN
- Slower to learn but provided stable results.
- Epsilon-greedy exploration worked well for this environment but may struggle in more complex setups.

### A2C
- Learned faster and achieved higher rewards earlier in training.
- Required more computational resources and was less stable during longer training sessions.


### TensorBoard Logs
![image](https://github.com/user-attachments/assets/3a356efc-7ed7-4956-89a0-258888f7646f)

![plot2](https://github.com/user-attachments/assets/18135995-d53d-4df3-aadd-e883c7bbed08)

---

## Future Plans

- Implement prioritized experience replay with DQN to improve exploration.
- Explore moving the replay buffer to a NoSQL database for better memory management.
- Investigate distributed training to reduce CPU bottlenecks and fully utilize GPU resources.

---

## Summary

This project deepened my understanding of Reinforcement Learning in practice. Both DQN and A2C demonstrated strengths, but each required unique approaches to training and resource optimization. While RL is technically demanding, overcoming these challenges provided valuable insights and practical experience.
