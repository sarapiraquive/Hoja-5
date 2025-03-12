from ._base import PolicyImprover
import numpy as np


class StandardPolicyImprover(PolicyImprover):

    def __init__(self, min_advantage=10**-15):
        """
            :param min_advantage: minimum improvement that a q-value must offer over the current state value to trigger a change in policy
        """
        self.min_advantage = min_advantage
        self._policy = {}

    def improve(self, q):
    
        """
            :param q: a 2-depth dictionary where q[s][a] = q(s,a)
            :return: True if the policy has been changed, False if not
            
            improves the policy based on current estimates of q accessible
        """
        policy_stable = True
        for s, actions in q.items():
            if not actions:
                print(f"Warning: Empty actions dictionary for state {s}") #debugging
                continue
            
            try:
                best_action = max(actions, key=actions.get)
                if s not in self._policy or self._policy[s] != best_action:
                    self._policy[s] = best_action
                    policy_stable = False
            except ValueError as e:
                print(f"Error trying to improve policy for state {s}: {e}")
        return not policy_stable
     
    @property
    def policy(self):
        return lambda s: self._policy.get(s, None)
