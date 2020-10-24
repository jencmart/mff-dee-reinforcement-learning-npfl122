__### Assignment: importance_sampling
#### Date: Deadline: Nov 03, 23:59
#### Points: 2 points
#### Examples: importance_sampling_examples

Using the [FrozenLake-v0 environment](https://gym.openai.com/envs/FrozenLake-v0)
environment, implement Monte Carlo weighted importance sampling to estimate
state value function $V$ of target policy, which uniformly chooses either action
1 (down) or action 2 (right), utilizing behaviour policy, which uniformly
chooses among all four actions.

Start with the [importance_sampling.py](https://github.com/ufal/npfl122/tree/master/labs/03/importance_sampling.py)
template, which creates the environment and generates episodes according to
behaviour policy.

#### Examples Start: importance_sampling_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 importance_sampling.py --episodes=500`
```
 0.00  0.00  0.00  0.00
 0.03  0.00  0.00  0.00
 0.22  0.14  0.29  0.00
 0.00  0.50  1.00  0.00
```
- `python3 importance_sampling.py --episodes=5000`
```
 0.00  0.01  0.02  0.00
 0.00  0.00  0.08  0.00
 0.06  0.08  0.17  0.00
 0.00  0.19  0.89  0.00
```
- `python3 importance_sampling.py --episodes=50000`
```
 0.02  0.01  0.04  0.01
 0.03  0.00  0.06  0.00
 0.08  0.17  0.24  0.00
 0.00  0.34  0.78  0.00
```
#### Examples End:
