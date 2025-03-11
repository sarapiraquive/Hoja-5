from ._base import PolicyEvaluator
import numpy as np
from mdp import get_closed_form_of_mdp


class LinearSystemEvaluator(PolicyEvaluator):

    def __init__(self, mdp, gamma):
    	pass # your code

    def _after_reset(self):
        """
            Update q-values
        """
    	pass # your code

    @property
    def provides_state_values(self):
        return True

    @property
    def v(self):
        return self._v_values
    
    @property
    def q(self):
        return self._q_values
