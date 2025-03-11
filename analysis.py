import matplotlib.pyplot as plt
import pandas as pd


class Analyzer:

    def __init__(self, mdp):
        """
            :param mdp: The MDP that is being analyzed
        """

        pass # your code
    
    def new_run(self, name):
        """
            :param name: The name of the run, used for figures
            
            indicates the analyzer that upcoming observations are for a new run of the policy learner
        """
        pass # your code
    
    def add_state_value_estimates(self, v):
        """
            :param v: dictionary with state values or estimates thereof
            
            add the state value estimates to the history of the current run
        """
        pass # your code
    
    def plot_state_value_estimates_of_init_state_over_time(self, ax=None):
        """
            :param ax: optional axis object where the lines are drawn. If not given, a new ax object is created

            Creates a line plot (step function) that shows one line for each run. The line value corresponds to the state value of the initial state of the MDP
        """
        pass # your code

    def plot_avg_state_value_estimates_of_init_state_over_time(self, ax=None):
        """
            :param ax: optional axis object where the lines are drawn. If not given, a new ax object is created

            Creates a line plot (step function) that shows one line for each run. The line value corresponds to the average value across all states in the MDP
        """
        pass # your code

