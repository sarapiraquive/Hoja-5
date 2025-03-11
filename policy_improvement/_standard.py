from ._base import PolicyImprover
import numpy as np


class StandardPolicyImprover(PolicyImprover):

    def __init__(self, min_advantage=10**-15):
        """
            :param min_advantage: minimum improvement that a q-value must offer over the current state value to trigger a change in policy
        """
        pass # your code

    def improve(self, q):
        pass # your code
    
    @property
    def policy(self):
        pass # your code
