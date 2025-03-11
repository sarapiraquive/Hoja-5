from ._base import PolicyEvaluator
import numpy as np
from mdp import get_closed_form_of_mdp


class LinearSystemEvaluator(PolicyEvaluator):

    def __init__(self, mdp, gamma):
        super().__init__(gamma)
        self.mdp = mdp
        self.states, self.probs, self.rewards = get_closed_form_of_mdp(mdp)
        self.n = len(self.states)
        self._v_values = {s: 0 for s in self.states}  # Inicializar valores de estado en 0


    def _after_reset(self):
        """
            Update q-values
        """
        gamma_adj = min(self.gamma, 0.9999)

        # Crear la matriz A y el vector y
        A = np.eye(self.n)  # Matriz identidad de tamaño n
        y = np.zeros(self.n)  # Vector de términos independientes

        for i, s in enumerate(self.states):
            A[i, i] -= gamma_adj * sum(self.probs.get(s, {}).get(self.policy(s), {}).get(s_prime, 0)
                                       for s_prime in self.states)
            y[i] = self.rewards[i]
            for j, s_prime in enumerate(self.states):
                A[i, j] += gamma_adj * self.probs.get(s, {}).get(self.policy(s), {}).get(s_prime, 0)

        v_values = np.linalg.solve(A, y)
        self._v_values = {s: v_values[i] for i, s in enumerate(self.states)}


    @property
    def provides_state_values(self):
        return True

    @property
    def v(self):
        return self._v_values
    
    @property
    def q(self):
        """
            calcula Q(s,a) a partir de v(s)
        """
        q_values = {}
        for s in self.states:
            q_values[s] = {}
            for a in self.mdp.get_actions_in_state(s):
                q_values[s][a] = self.rewards[self.states.index(s)] + self.gamma * sum(
                    self.probs[s][a].get(s_prime, 0) * self._v_values[s_prime] for s_prime in self.states
                )
        return q_values

