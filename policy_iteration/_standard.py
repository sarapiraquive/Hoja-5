from ._base import PolicyIteration


class StandardPolicyIteration(PolicyIteration):

    def __init__(self, init_policy, policy_evaluator, policy_improver):
        """
            :param init_policy: policy with which the algorithm is initialized
            :param policy_evaluator: the policy evaluator
            :param policy_improver: the policy improver
        """
        self.policy_evaluator = policy_evaluator
        self.policy_improver = policy_improver
        self.policy_evaluator = policy_evaluator
    
    def step(self):
        q_values = self.policy_evaluator.q
        improved = self.policy_improver.improve(q_values)
        self.policy_evaluator.reset(self.policy_improver.policy)
        return improved
