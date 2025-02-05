from main import find_sigma_labeling
import networkx as nx

# Create a bipartite graph
graph = nx.bipartite.random_graph(3, 3, 5)

# Find a σ^{+-}-labeling
labeled_graph = find_sigma_labeling(graph)

# Print the labeled graph
print("Labeled Graph:")
print(nx.get_node_attributes(labeled_graph, "label"))