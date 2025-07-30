# Rwanda Textbook Distribution RL System

## Project Overview

This project implements a reinforcement learning system to optimize textbook distribution decisions in Rwanda's education system. The system simulates a smart decision-making agent that evaluates school conditions and determines the most effective textbook delivery actions to maximize learning outcomes while minimizing waste.

### Problem Context

Based on Rwanda's Foundational Learning Strategy (FLS), many schools face challenges in textbook access due to:
- Insufficient textbook-to-student ratios
- Poor coordination between government grants and actual needs  
- Mismatched textbook quality and curriculum requirements
- Inefficient delivery logistics between rural and urban schools

### Solution Approach

The RL agent operates within a custom Gymnasium environment where:
- **States**: School characteristics (student count, available textbooks, teacher guides, grant usage, urgency level, location type, infrastructure, quality scores, delivery history)
- **Actions**: Delivery decisions (send textbooks, hold delivery, reassign batch, send guides, send limited supply, flag for follow-up)
- **Rewards**: Based on improved textbook-to-student ratios, appropriate material matching, effective usage, and addressing high-urgency situations

## Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd rwanda-textbook-rl

# Install dependencies
pip install -r requirements.txt

# Run experiments
python main.py --algorithm dqn --experiment 1
python main.py --list-experiments  # See all available experiments
```

## Usage

### Running Individual Experiments

```bash
# DQN experiments (1-4 available)
python main.py --algorithm dqn --experiment 1

# PPO experiments (1-3 available)  
python main.py --algorithm ppo --experiment 2

# REINFORCE experiments (1-3 available)
python main.py --algorithm reinforce --experiment 1

# Actor-Critic experiments (1-3 available)
python main.py --algorithm actor_critic --experiment 3
```

### Running 3D Visualization
```bash
# reload page if it show an error.
python -m renderer.start_visualization
```

### Project Structure

```
├── environment/
│   ├── custom_env.py              # Main discrete environment
│   ├── custom_env_continuous.py   # Continuous action environment
│   └── rendering.py               # 3D visualization components
├── train/
│   ├── dqn_training.py            # DQN training script
│   ├── ppo_training.py            # PPO training script
│   ├── reinforce_training.py      # REINFORCE training script
│   └── actor_critic_training.py   # Actor-Critic training script
├── model/
│   ├── dqn/                       # Saved DQN models
│   └── pg/                        # Saved policy gradient models
├── logs/                          # Training logs and metrics
├── main.py                        # Main entry point
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Experimental Results and Analysis before rendering trained agent

### DQN Experiment Summary Table

| Experiment | Key Hyperparameters | Final Mean Reward | Best Mean Reward | Final Ep. Length | Best Ep. Length | Convergence | Notes |
|------------|---------------------|-------------------|------------------|------------------|-----------------|-------------|-------|
| DQN #1     | lr=0.001, γ=0.99, batch=32, eps=1.0→0.02, expl_frac=0.1, buf=10k | 38,000 | 40,000 | 122 | 142 | Partial | Peak at 60k steps, then unstable. High learning rate may cause instability. |
| DQN #2     | lr=0.0005, γ=0.99, batch=32, eps=1.0→0.02, expl_frac=0.1, buf=50k | 31,000 | 35,600 | 111 | 134 | Partial | Late surge after 80k steps, then drop. Better stability than #1. |
| DQN #3     | lr=0.0005, γ=0.99, batch=64, eps=1.0→0.02, expl_frac=0.3, buf=50k | 4,521 | 7,693 | 4.7 | 6.5 | No | Highly unstable, failed to learn. Larger batch + more exploration hurt performance. |
| DQN #4     | lr=0.00025, γ=0.95, batch=64, eps=1.0→0.02, expl_frac=0.2, buf=100k | 5,871 | 7,609 | 7.4 | 10.0 | No | Better than #3 but still unstable. Conservative settings didn't help enough. |

### DQN Detailed Analysis

**DQN Experiment 1:** Used a learning rate of 0.001, gamma 0.99, batch size 32, and replay buffer of 10,000. Agent started with full exploration (epsilon=1.0) and decayed to 0.02 over the first 10% of training. Mean reward increased steadily, peaking around 40,000, but then dropped and fluctuated before recovering by the end. Episode length followed a similar pattern. The agent did not fully converge within 100,000 timesteps, as the reward curve never fully flattened. The learning rate may be too high or the buffer too small, and more exploration or a longer training period could help.

**DQN Experiment 2:** Used a lower learning rate (0.0005) and a larger replay buffer (50,000) compared to Experiment 1. The mean reward stayed flat for most of training, then rapidly increased after 80,000 timesteps, peaking at 35,600. However, there was a significant drop and only a partial recovery near the end, so the agent did not fully converge. Episode length followed a similar late surge and drop. The changes to learning rate and buffer size helped the agent learn more stably than Experiment 1, but the final and best mean rewards were lower. When considering mean reward, Experiment 1 performed better overall, but experiment 2 maintains the best balance of reward and stability.

**DQN Experiment 3:** Used a larger batch size (64) and longer exploration fraction (0.3) with the same learning rate and buffer as DQN #2. The mean reward was low and highly unstable throughout training, fluctuating between 3,000 and 7,700, with no clear upward trend or convergence. Episode lengths were very short and also unstable. This suggests the agent failed to learn an effective policy. Compared to DQN #1 and #2, this experiment performed much worse, likely because the larger batch size and longer exploration did not help and may have hindered learning in this environment.

**DQN Experiment 4:** Used a lower learning rate (0.00025), lower gamma (0.95), larger batch size (64), and a much larger buffer (100,000). The mean reward was slightly higher than DQN #3, peaking at 7,609, but remained unstable throughout training with no clear convergence. Episode lengths were longer but also unstable. This experiment performed better than DQN #3, but was still much worse than DQN #1 and #2. DQN #2 remains the best balance of reward and stability, while DQN #1 had the highest reward but less stability.


## PPO Experiment Results and Analysis

### PPO Experiment Summary Table


| Experiment | Key Hyperparameters                          | Final Mean Reward | Best Mean Reward | Final Ep. Length | Best Ep. Length | Convergence | Training Stability | Notes                                                                 |
|------------|---------------------------------------------|-------------------|------------------|------------------|-----------------|-------------|-------------------|----------------------------------------------------------------------|
| PPO #1     | lr=0.0001, γ=1.0, n_steps=4096, batch=128, ent_coef=0.0, clip=0.1, epochs=10, net=[64,64] | 11,236            | 13,688           | 99.98            | 100.0           | Partial     | Good              | Conservative exploitation-focused settings. Shows learning but plateaus after 33k steps. |
| PPO #2     | lr=0.0003, γ=1.0, n_steps=2048, batch=128, ent_coef=0.01, clip=0.2, epochs=10, net=[64,64] | 13,190.22         | 16,074.91        | 100.00           | 100.00          | Partial     | Moderate          | Exploration-focused settings. Higher peak reward but unstable with a downward trend late in training. |
| PPO #3     | lr=0.0002, γ=1.0, n_steps=4096, batch=128, ent_coef=0.02, clip=0.25, epochs=10, net=[64,64] | 16,852.97         | 16,852.97        | 100.00           | 100.00          | Good        | Good              | Fine-tuned settings. Strong upward reward trend with stable episode lengths, overcoming prior instability. |

### PPO Detailed Analysis

**PPO Experiment 1:** The agent got a large reward at the beginning of the timestep which is understandable because the hyperparamaters where designed to to do more of exploitation, but around ~16k timesteps the model was training relatively stable with no catastrophic drops, after the first ~16k steps indicating the environment is learnable with exploitation-focused learning, but the plateau from ~33k to ~100k steps where it seems the agent can't acheive a reward or episode (Final mean reward: 13995.48, Final mean episode length: 99.62) better than the inital step suggests the agent may be stuck in a local optimum or needs more exploration, also taking a look at the episode per length shows something interesting due to the fact that exploitation was more emphasized in the first experiment you would notice a downward trend in the episode length that as the steps or training gets longer because of the emphasis on exploitation the agent perform poorly as the episode descrease, so in our next experiment we would try exploring better and see how the model performs.

**PPO Experiment 2:**
The agent showed a promising upward trend in mean rewards, reaching a peak of 16,074.91 around the 82,000-timestep mark, though a slight downward trend emerged in the final timesteps, resulting in a final mean reward of 13,190.22. Episode lengths initially decreased over the first 40,000 timesteps, reflecting early exploration, before trending upward and stabilizing at the maximum episode length of 100.00 from approximately 50,000 timesteps to the end, indicating the agent adapted to sustain full episodes. Compared to PPO Experiment 1, which achieved a final mean reward of 13,995.48 and a best mean reward of 15,028.56 over 102,000 timesteps with a final episode length of 99.62, Experiment 2 outperformed in peak reward (6.9% higher) and episode length consistency (100.00 vs. 99.62), but underperformed in final reward stability (5.8% lower), suggesting the increased exploration helped discover a better policy but introduced late-stage instability. The initial reward drop and final reward decline highlight a potential over-adjustment, prompting the next experiment to focus on stabilizing the gains

**PPO Experiment 3:**
The agent exhibited a clear upward trend in mean rewards, culminating in a final mean reward of 16,852.97 and a best mean reward of 16,852.97 over 102,000 timesteps, marking a significant improvement over prior experiments. Episode lengths followed a pattern similar to PPO Experiment 2 initially, with a sharp dip early on, followed by an upward trend that flattened out at the maximum episode length of 100.00, indicating the agent consistently sustained full episodes. Compared to PPO Experiment 2, which achieved a final mean reward of 13,190.22 and a best mean reward of 16,074.91 with a final episode length of 100.00, Experiment 3 outperformed both in final reward (27.7% higher) and matched the best reward improvement (5.0% higher), while maintaining episode length stability. The upward reward trend and lack of a late downward shift suggest that the adjusted learning_rate, increased clip_range and entrophy coefficient effectively enhanced learning without the instability seen in Experiment 2, though the sharp initial dip in episode length hints at early exploration challenges that the agent overcame, pointing to a robust policy development.


## Reinforce Experiment Results and Analysis

### Reinforce Experiment Summary Table


| Experiment | Total Timesteps | Final Mean Reward | Best Mean Reward | Final Ep. Length | Best Ep. Length | Learning Rate | Gamma | Update Threshold | Hidden Units | Observations |
|------------|-----------------|-------------------|------------------|------------------|-----------------|---------------|-------|-----------------|-------------|-------------|
| 1          | 200,000         | 7,335.98          | 7,706.22         | 4.54             | 5.19            | 0.001         | 0.99  | 100             | 64          | Noisy, wave-like reward trend with short episodes, suggesting high learning rate instability. |
| 2          | 200,000         | 4,432.63          | 8,654.18         | 4.68             | 5.42            | 0.0005        | 1.0   | 1000            | 128         | Early high reward peak with short episodes, followed by decline, indicating limited adaptation. |
| 3          | 200,000         | 4,315.97          | 7,339.42         | 12.33            | 12.64           | 0.005         | 1.0   | 1000            | 64          | Initial reward rise with short episodes, then decline with longer episodes, suggesting early termination. |

### Reinforce Detailed Analysis

**Reinforce Experiment 1:** The agent showed a noisy, wave-like trend in mean rewards over 200,000 timesteps, peaking at 7,706.22 and ending at 7,335.98, with mean episode lengths oscillating between 4.54 and 5.19, well below the 1,000-step maximum. The close reward clustering and short episodes suggest a high learning rate (0.001) may cause overcorrections, and the 100-reward update threshold might not smooth the noise, indicating REINFORCE’s Monte Carlo approach struggles to stabilize.


**Reinforce Experiment 2:** For Training 2, with hyperparameters lr=0.0005, gamma=1.0, hidden_state=128, update_steps=1000, and max_steps=200000, the mean reward starts at 6,646.74, peaks at 8,654.18 around 17,000 timesteps, and declines to 4,432.63 by 200,000 timesteps, while mean episode length remains stable between 4.68 and 5.42, indicating limited exploration beyond quick delivery strategies; this lower learning rate compared to Experiment 1’s 0.001 enables a higher peak but the large update threshold (1000) and gamma=1.0 lead to a steady decline, suggesting overfitting to early rewards, but in comparison to experiment it does worse in terms of rewards, from their mean rewards graph the graph of experiment 1 seems consistent, but that of experiment of 2 peaks at the beginning then drops and doesn't get close to the best rewards indicating overfitting like I mentioned, but the mean episodes are almost alike so both experiment lacked poor exploration. 


**Reinforce Experiment 3:** For Training 3, with lr=0.005, gamma=1.0, hidden_state=64, update_steps=1000, and max_steps=200000, the mean reward rises from 4,571.22 to a peak of 7,339.42 around 41,000 timesteps before dropping to 4,315.97, with mean episode length increasing from 6.79 to 12.33, reflecting greater exploration; the higher learning rate causes volatility and overcorrections, with longer episodes emerging as rewards decline, possibly due to early termination strategies losing effectiveness, incomparison to experiment 2, expriment 3 did better than experiment 2 from the gaph the peaks were distributed along a higher range of time steps unlike experiment 2 which was short, and also it explored better but this exploration didn't make still help it maimize the rewards make it less better than experiment 2.


## Actor-Critic Experiment Results and Analysis


### Actor-Critic Experiment Summary Table
| Experiment | Total Timesteps | Final Mean Reward | Best Mean Reward | Final Ep. Length | Best Ep. Length | Learning Rate | Gamma | n_steps | GAE Lambda | VF Coef | Max Grad Norm | Hidden Layers | Observations |
|------------|-----------------|-------------------|------------------|------------------|-----------------|---------------|-------|---------|------------|---------|--------------|--------------|-------------|
| 1          | 200,000         | 12,528.31         | 15,660.66         | 99.75             | 100.00          | 0.0001        | 1.0   | 4096     | 0.95       | 0.5     | 0.5          | [64, 64]     | High initial rewards followed by a decline and wave-like stability, suggesting early learning strength but limited long-term adaptation. |
| 2          | 200,000         | 13,493.30         | 16,873.36         | 100.00            | 100.00          | 0.0003        | 1.0   | 2048     | 0.95       | 0.5     | 0.5          | [64, 64]     | Strong initial performance with a gradual decline to a stable state resembling Experiment 1's but it is a bit better as experiment 1 plateaued but this due to exploration still got to learn and maximmize reward evident as the final mean reward is better. |
| 3          | 200,000         | 13,735.67         | 14,411.99         | 100.00            | 100.00          | 0.0002        | 1.0   | 4096     | 0.95       | 0.5     | 0.5          | [64, 64]     | Low initial rewards with steady improvement, peaking later and stabilizing in a wave-like pattern, reflecting effective exploration and stable learning. |

### Actor-Critic Detailed Analysis

**Actor-Critic Experiment 1:** The agent began with a mean reward of 15,660.66 at 1,000 timesteps, declining to 12,528.31 by 200,000 timesteps, with mean episode lengths consistently near 100.0, occasionally dropping to 99.75. With a learning rate of 0.0001, a high n_steps of 4096, and GAE lambda of 0.95, the initial high reward suggests strong early learning. However, the subsequent decline indicates potential convergence issues, possibly due to a low learning rate limiting adaptation or the large n_steps causing delayed updates that hinder optimization


**Actor-Critic Experiment 2:** The agent achieved a peak mean reward of 16,873.36 at 1,000 timesteps, rising steadily to 13,493.30 by 200,000 timesteps, with mean episode lengths stable at 100.0. Configured with a learning rate of 0.0003, n_steps of 2048, and GAE lambda of 0.95, this setup demonstrates robust learning and adaptation. The higher learning rate compared to Experiment 1 likely facilitated better exploration, while the smaller n_steps allowed more frequent updates, contributing to consistent performance throughout training.


**Actor-Critic Experiment 3:** The agent began with a low mean reward of 8,059.42 at 1,000 timesteps, improving steadily to a peak of 14,411.99 around 149,000 timesteps, and settling at 13,735.67 by 200,000 timesteps, with mean episode lengths consistently at 100.0. With a learning rate of 0.0002, n_steps of 4096, and GAE lambda of 0.95, this configuration enabled effective exploration, as evidenced by the gradual reward increase. The later wave-like pattern with peaks and troughs, similar to Experiment 1, suggests stable learning after reaching a performance plateau, making it the most balanced approach with sustained improvement and adaptability.


## Overall Algorithm Comparison

Based on the experimental results, **PPO Experiment #3** achieved the best overall performance with a final mean reward of 16,852.97 and stable episode lengths of 100.0. The ranking of algorithms by peak performance:

1. **PPO #3**: 16,852.97 (best overall - stable learning with upward trend)
2. **Actor-Critic #2**: 16,873.36 (highest peak but declined significantly)  
3. **DQN #1**: 40,000 (highest absolute but highly unstable)
4. **Actor-Critic #3**: 14,411.99 (most stable learning curve)
5. **PPO #2**: 16,074.91 (good peak but late degradation)

PPO #3 stands out as the most reliable algorithm for this textbook di


## **Recommendations for Improvement after rendering trained agent**:
- Revise the reward function to include imbalance penalties and reward incremental progress.
- Shift to a continuous action space for finer control.
- Adjust termination to include fixed step limits or “good enough” states.
- Increase exploration (e.g., entropy regularization, higher epsilon) to prevent fixation.
- Extend training to 500,000 timesteps for better policy learning.