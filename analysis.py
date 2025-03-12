import matplotlib.pyplot as plt
import pandas as pd


class Analyzer:

    def __init__(self, mdp):
        """
            :param mdp: The MDP that is being analyzed
        """

        self.mdp = mdp
        self.runs = {}
        self.current_run = None
    
    def new_run(self, name):
        """
            :param name: The name of the run, used for figures
            
            indicates the analyzer that upcoming observations are for a new run of the policy learner
        """
        self.current_run = name
        self.runs[name] = []
    
    def add_state_value_estimates(self, v):
        """
            :param v: dictionary with state values or estimates thereof
            
            add the state value estimates to the history of the current run
        """
        if self.current_run is not None:
            self.runs[self.current_run].append(v.copy())
    
    def plot_state_value_estimates_of_init_state_over_time(self, ax=None):
        """
            :param ax: optional axis object where the lines are drawn. If not given, a new ax object is created

            Creates a line plot (step function) that shows one line for each run. The line value corresponds to the state value of the initial state of the MDP
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        for run_name, history in self.runs.items():
            values = [v[self.mdp.init_states[0]] for v in history]
            ax.step(range(len(values)), values, label=run_name)
        
        ax.set_xlabel("Iteration t")
        ax.set_ylabel("Value of Initial State")
        ax.legend()
        plt.show()

    def plot_avg_state_value_estimates_of_init_state_over_time(self, ax=None):
        """
            :param ax: optional axis object where the lines are drawn. If not given, a new ax object is created

            Creates a line plot (step function) that shows one line for each run. The line value corresponds to the average value across all states in the MDP
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        for run_name, history in self.runs.items():
            avg_values = [sum(v.values()) / len(v) for v in history]
            ax.step(range(len(avg_values)), avg_values, label=run_name)
        
        ax.set_xlabel("Iteration t")
        ax.set_ylabel("Average State Value")
        ax.legend()
        plt.show()

