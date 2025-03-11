from ._base import PolicyIteration


class StandardPolicyIteration(PolicyIteration):

    def __init__(self, init_policy, policy_evaluator, policy_improver):
        """
            :param init_policy: policy with which the algorithm is initialized
            :param policy_evaluator: the policy evaluator
            :param policy_improver: the policy improver
        """
        pass # your code
    
    def step(self):
        pass # your code
