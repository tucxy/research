import numpy as np
import networkx as nx
from gymnasium import spaces, Env

class GraphLabelingEnv(Env):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.m_edges = graph.number_of_edges()
        self.observation_space = spaces.Box(low=-1, high=2*self.m_edges - 2, shape=(self.n_nodes,), dtype=np.int32)
        self.action_space = spaces.Discrete(2 * self.m_edges - 1)

        self.state = np.full(self.n_nodes, -1, dtype=np.int32)
        self.current_node_idx = 0

    def reset(self, seed=None, options=None):
        self.state = np.full(self.n_nodes, -1, dtype=np.int32)
        self.current_node_idx = 0
        return self.state, {}

    def step(self, action):
        self.state[self.current_node_idx] = action
        reward = self._calculate_reward()
        done = self.current_node_idx == self.n_nodes - 1
        if not done:
            self.current_node_idx += 1
        return self.state, reward, done, False, {}

    def _calculate_reward(self):
        reward = 0
        labeled_nodes = np.where(self.state != -1)[0]
        labels = self.state[labeled_nodes]
        
        # Penalize duplicates
        unique_labels, counts = np.unique(labels, return_counts=True)
        reward -= 100 * (len(labels) - len(unique_labels))
        
        # Edge constraints
        for u, v in self.graph.edges():
            if u in labeled_nodes and v in labeled_nodes:
                label_u = self.state[u]
                label_v = self.state[v]
                if label_u < label_v:
                    reward += 1
                elif label_u > label_v:
                    reward -= 1
                if abs(label_u - label_v) == self.m_edges:
                    reward -= 10
        return reward

    def get_labeled_graph(self):
        labeled_graph = self.graph.copy()
        labels = {node: int(self.state[i]) for i, node in enumerate(self.graph.nodes())}
        nx.set_node_attributes(labeled_graph, labels, "label")
        return labeled_graph