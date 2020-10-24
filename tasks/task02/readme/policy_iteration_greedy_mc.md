### Assignment: policy_iteration_greedy_mc
#### Date: Deadline: Oct 27, 23:59
#### Points: 2 points
#### Examples: policy_iteration_greedy_mc_examples
    # evaluation, estimate action-value function by Monte Carlo simulation:
    # - for state in range(env.states):
    #   - start in a given state
    #   - perform `args.mc_length` Monte Carlo steps, utilizing
    #     epsilon-greedy actions with respect to the policy, using
    #     `env.epsilon_greedy(args.epsilon, greedy_action)`
    #     - this metod returns a random action with probability `args.epsilon`
    #     - otherwise it returns the passed `greedy_action`
    #     - for replicability, make sure to call it exactly `args.mc_length`
    #       times in every simulation
    #   - compute the return of the simulation
    #   - update the estimate using averaging, but only for the first
    #     occurrence of the first state-action pair
    
    
    # During the policy improvement, if multiple actions have the same estimate,
    # choose the one with the smaller index.


Starting with [policy_iteration_greedy_mc.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration_greedy_mc.py),
extend the `policy_iteration_exploring_mc` assignment to perform policy
evaluation by using $ε$-greedy Monte Carlo estimation.

For the sake of replicability, use the provided
`GridWorld.epsilon_greedy(epsilon, greedy_action)` method, which returns
a random action with probability of `epsilon` and otherwise returns the
given `greedy_action`.

#### Examples Start: policy_iteration_greedy_mc_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00→    0.00→
    0.00↑    0.00↑    0.00→    0.00→
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=10`
```
   -1.20↓   -1.43←    0.00←   -6.00↑
    0.78→           -20.26↓    0.00←
    0.09←    0.00↓   -9.80↓   10.37↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=50`
```
   -0.16↓   -0.19←    0.56←   -6.30↑
    0.13→            -6.99↓   -3.51↓
    0.01←    0.00←    3.18↓    7.57↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=100`
```
   -0.07↓   -0.09←    0.28←   -4.66↑
    0.06→            -5.04↓   -8.32↓
    0.00←    0.00←    1.70↓    4.38↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=200`
```
   -0.04↓   -0.04←   -0.76←   -4.15↑
    0.03→            -8.02↓   -5.96↓
    0.00←    0.00←    2.53↓    4.36↓
```
- `python3 policy_iteration_greedy_mc.py --gamma=0.95 --seed=42 --steps=500`
```
   -0.02↓   -0.02←   -0.65←   -3.52↑
    0.01→           -11.34↓   -8.07↓
    0.00←    0.00←    3.15↓    3.99↓
```


#### Examples End:





