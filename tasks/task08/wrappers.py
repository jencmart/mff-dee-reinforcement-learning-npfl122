#!/usr/bin/env python3
import sys

import gym
import numpy as np

############################
# Gym Environment Wrappers #
############################

class EvaluationWrapper(gym.Wrapper):
    def __init__(self, env, seed=None, evaluate_for=100, report_each=10):
        super().__init__(env)
        self._evaluate_for = evaluate_for
        self._report_each = report_each

        self.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self._episode_running = False
        self._episode_returns = []
        self._evaluating_from = None
        self.ENV = env

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, X=(False, None)):
        was_tupe = True
        if isinstance(X, tuple):
            start_evaluation = X[0]
            states = X[1]
        else:
            start_evaluation = X
            states = None
            was_tupe = False
        if self._evaluating_from is not None and self._episode_running:
            raise RuntimeError("Cannot reset a running episode after `start_evaluation=True`")

        if start_evaluation and self._evaluating_from is None:
            self._evaluating_from = self.episode

        self._episode_running = True
        self._episode_return = 0
        self.env._elapsed_steps = 0
        if not start_evaluation and was_tupe and isinstance(states, tuple):
            return self.ENV.reseset(states)
        else:
            return super().reset()

        # return super().reset()

    def step(self, action):
        if not self._episode_running:
            raise RuntimeError("Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, done, info = super().step(action) # z(action)

        self._episode_return += reward
        if done:
            self._episode_running = False
            self._episode_returns.append(self._episode_return)

            if self._report_each and self.episode % self._report_each == 0:
                print("Episode {}, mean {}-episode return {:.2f} +-{:.2f}".format(
                    self.episode, self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:])), file=sys.stderr)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + self._evaluate_for:
                print("The mean {}-episode return after evaluation {:.2f} +-{:.2f}".format(
                    self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:]), file=sys.stderr))
                self.close()
                # sys.exit(0)

        return observation, reward, done, info


# Utilizites #
##############
def typed_np_function(*types):
    """Typed NumPy function decorator.

    Can be used to wrap a function expecting NumPy inputs.

    It converts input positional arguments to NumPy arrays of the given types,
    and passes the result through `np.array` before returning (while keeping
    original tuples, lists and dictionaries).
    """
    def check_typed_np_function(wrapped, args):
        if len(types) != len(args):
            while hasattr(wrapped, "__wrapped__"): wrapped = wrapped.__wrapped__
            raise AssertionError("The typed_np_function decorator for {} expected {} arguments, but got {}".format(wrapped, len(types), len(args)))

    def structural_map(function, value):
        if isinstance(value, tuple):
            return tuple(structural_map(function, element) for element in value)
        if isinstance(value, list):
            return [structural_map(function, element) for element in value]
        if isinstance(value, dict):
            return {key: structural_map(function, element) for key, element in value.items()}
        return function(value)

    class TypedNpFunctionWrapperMethod:
        def __init__(self, instance, func):
            self._instance, self.__wrapped__ = instance, func
        def __call__(self, *args, **kwargs):
            check_typed_np_function(self.__wrapped__, args)
            return structural_map(np.array, self.__wrapped__(*[np.asarray(arg, typ) for arg, typ in zip(args, types)], **kwargs))

    class TypedNpFunctionWrapper:
        def __init__(self, func):
            self.__wrapped__ = func
        def __call__(self, *args, **kwargs):
            check_typed_np_function(self.__wrapped__, args)
            return structural_map(np.array, self.__wrapped__(*[np.asarray(arg, typ) for arg, typ in zip(args, types)], **kwargs))
        def __get__(self, instance, cls):
            return TypedNpFunctionWrapperMethod(instance, self.__wrapped__.__get__(instance, cls))

    return TypedNpFunctionWrapper
