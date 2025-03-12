from ._base import PolicyImprover
import numpy as np


class StandardPolicyImprover(PolicyImprover):

    def __init__(self, min_advantage=10**-15):
        """
            :param min_advantage: minimum improvement that a q-value must offer over the current state value to trigger a change in policy
        """
        self.min_advantage = min_advantage
        self.policy = {}

    def improve(self, q):
        policy_stable = True
        for s, actions in q.items():
            best_action = max(actions, key=actions.get)
            if s not in self.policy or self._policy[s] != best_action:
                self.policy[s] = best_action
                policy_stable = False
        return not policy_stable
    
    @property
    def policy(self):
        return lambda s: self._policy.get(s, None)
