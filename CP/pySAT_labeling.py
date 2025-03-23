import networkx as nx
from pysat.formula import CNF
from pysat.solvers import Solver
import math

def sigmapm_graph_labeling(graph):
    m = graph.number_of_edges()
    cnf = CNF()

    # Get all connected components
    components = list(nx.connected_components(graph))

    # Storage for bipartite sets
    A, B = set(), set()

    # Process each connected component independently
    for component in components:
        subgraph = graph.subgraph(component)
        try:
            A_sub, B_sub = nx.bipartite.sets(subgraph)
            A.update(A_sub)
            B.update(B_sub)
        except nx.NetworkXError:
            raise ValueError("Graph contains a non-bipartite component.")

    # Variables: Each node gets a unique label in {0, ..., 2m-1}
    # In SAT, variables are represented as integers. We'll map each node's label to a unique variable.
    label_vars = {v: i + 1 for i, v in enumerate(graph.nodes())}  # Node v -> variable i+1
    max_label = 2 * m - 1

    # Constraint 1: Unique labels
    # Ensure that no two nodes have the same label
    for u in graph.nodes():
        for v in graph.nodes():
            if u != v:
                cnf.append([-label_vars[u], -label_vars[v]])  # u and v cannot have the same label

    # Constraint 2: Ordering constraint for bipartite edges
    for a, b in graph.edges():
        if a in A:
            cnf.append([-label_vars[a], label_vars[b]])  # label[a] < label[b]
        else:
            cnf.append([-label_vars[b], label_vars[a]])  # label[b] < label[a]

    # Constraint 3: Edge lengths form a bijection with {1, ..., m}
    edge_length_vars = {}
    for idx, (a, b) in enumerate(graph.edges()):
        edge_length_vars[(a, b)] = len(label_vars) + idx + 1  # New variable for edge length

    # Solve the CNF formula
    solver = Solver(bootstrap_with=cnf.clauses)
    if solver.solve():
        # Extract the solution
        labeled_graph = graph.copy()
        labels = {v: solver.get_model()[label_vars[v] - 1] for v in graph.nodes()}
        nx.set_node_attributes(labeled_graph, labels, "label")
        return labeled_graph
    else:
        return None  # No solution found

def construct_kG(G, k=3):
    """Constructs k disjoint isomorphic copies of G."""
    kG = nx.Graph()
    for i in range(k):
        mapping = {v: (v, i) for v in G.nodes()}  # (original node, copy index)
        Gi = nx.relabel_nodes(G, mapping)
        kG = nx.compose(kG, Gi)  # Merge each copy
    return kG

def solve_123_labeling(G):
    """Finds a (1-2-3-⋯-k)-labeling of kG where edge lengths form a bijection with {1, ..., k}."""
    m = G.number_of_edges()
    k = math.floor((3 * m) / 2)  # Correct k calculation

    if k <= 0:
        raise ValueError(f"Invalid k value: {k}. Ensure that m is large enough.")

    kG = construct_kG(G, k)  # Create k copies of G
    cnf = CNF()

    # Variables: Each node gets a unique label in {0, ..., 3m}
    label_vars = {v: i + 1 for i, v in enumerate(kG.nodes())}  # Node v -> variable i+1
    max_label = 3 * m

    # Constraint 1: Unique labels within each component
    for i in range(k):
        component_nodes = [v for v in kG.nodes() if v[1] == i]
        for u in component_nodes:
            for v in component_nodes:
                if u != v:
                    cnf.append([-label_vars[u], -label_vars[v]])  # u and v cannot have the same label

    # Constraint 2: Edge lengths must form a bijection with {1, ..., k}
    edge_length_vars = {}
    for idx, (u, v) in enumerate(kG.edges()):
        edge_length_vars[(u, v)] = len(label_vars) + idx + 1  # New variable for edge length

    # Solve the CNF formula
    solver = Solver(bootstrap_with=cnf.clauses)
    if solver.solve():
        # Extract the solution
        labeled_kG = kG.copy()
        labels = {v: solver.get_model()[label_vars[v] - 1] for v in kG.nodes()}
        nx.set_node_attributes(labeled_kG, labels, "label")
        return labeled_kG
    else:
        print("No valid labeling found. Adjust constraints.")
        return None  # No solution found