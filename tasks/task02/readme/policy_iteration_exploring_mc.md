### Assignment: policy_iteration_exploring_mc
#### Date: Deadline: Oct 27, 23:59
#### Points: 2 points
#### Examples: policy_iteration_exploring_mc_examples

Starting with [policy_iteration_exploring_mc.py](https://github.com/ufal/npfl122/tree/master/labs/02/policy_iteration_exploring_mc.py),
extend the `policy_iteration` assignment to perform policy evaluation
by using Monte Carlo estimation with exploring starts.

The estimation can now be performed model-free (without the access to the full
MDP dynamics), therefore, the `GridWorld.step` returns a randomly sampled
result instead of a full distribution.

#### Examples Start: policy_iteration_exploring_mc_examples
Note that your results may sometimes be slightly different (for example because of varying floating point arithmetic on your CPU).
- `python3 policy_iteration_exploring_mc.py --gamma=0.95 --seed=42 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00↑    0.00↑
    0.00↑    0.00→    0.00↑    0.00↓
```
- `python3 policy_iteration_exploring_mc.py --gamma=0.95 --seed=42 --steps=10`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00↑  -19.50↑
    0.27↓    0.48←    2.21↓    8.52↓
```
- `python3 policy_iteration_exploring_mc.py --gamma=0.95 --seed=42 --steps=50`
```
    0.09↓    0.32↓    0.22←    0.15↑
    0.18↑            -2.43←   -5.12↓
    0.18↓    1.80↓    3.90↓    9.14↓
```
- `python3 policy_iteration_exploring_mc.py --gamma=0.95 --seed=42 --steps=100`
```
    3.09↓    2.42←    2.39←    1.17↑
    3.74↓             1.66←    0.18←
    3.92→    5.28→    7.16→   11.07↓
```
- `python3 policy_iteration_exploring_mc.py --gamma=0.95 --seed=42 --steps=200`
```
    7.71↓    6.76←    6.66←    3.92↑
    8.27↓             6.17←    5.31←
    8.88→   10.12→   11.36→   13.92↓
```
#### Examples End:
