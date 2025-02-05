from rl_agent import RLAgent
from genetic_algorithm import GeneticAlgorithm
import networkx as nx

class HybridSystem:
    def __init__(self, graph):
        self.graph = graph
        self.rl_agent = RLAgent(graph)
        self.ga = GeneticAlgorithm(graph, generations=200)  # Increase from 100 to 200

    def find_labeling(self):
        """Find a σ^{+-}-labeling using RL and GA."""
        # Step 1: Use RL to find an initial labeling
        print("Training RL agent...")
        self.rl_agent.train(total_timesteps=10000)
        rl_labeled_graph = self.rl_agent.predict()

        # Step 2: Use GA to refine the RL solution
        print("Running genetic algorithm...")
        initial_labeling = nx.get_node_attributes(rl_labeled_graph, "label")
        best_labeling = self.ga.evolve()

        # Return the best labeled graph
        return self.ga.get_labeled_graph(best_labeling)