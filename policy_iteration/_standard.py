from ._base import PolicyIteration

class StandardPolicyIteration(PolicyIteration):
    def __init__(self, init_policy, policy_evaluator, policy_improver):
        """
        :param init_policy: policy with which the algorithm is initialized
        :param policy_evaluator: the policy evaluator
        :param policy_improver: the policy improver
        """
        super().__init__(policy_evaluator, policy_improver)
        self.policy_evaluator.reset(init_policy)
    
    def step(self):
        """
        :return: True if the policy was improved, False otherwise
        """
        # Obtener los valores Q actuales
        q_values = self.policy_evaluator.q
        
        # Mejorar la política basada en los valores Q
        improved = self.policy_improver.improve(q_values)
        
        # Resetear el evaluador con la nueva política
        if improved:
            self.policy_evaluator.reset(self.policy_improver.policy)
        
        return improved