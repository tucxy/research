import networkx as nx
from ortools.sat.python import cp_model
import math

def sigmapm_graph_labeling(graph):
    model = cp_model.CpModel()
    m = graph.number_of_edges()
    
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
    labels = {v: model.NewIntVar(0, 2*m - 1, f"label_{v}") for v in graph.nodes()}
    
    # Constraint 1: Unique labels
    model.AddAllDifferent(labels.values())

    # Constraint 2: Ordering constraint for bipartite edges
    for a, b in graph.edges():
        if a in A:
            model.Add(labels[a] < labels[b])
        else:
            model.Add(labels[b] < labels[a])

    # Constraint 3: Edge lengths form a bijection with {1, ..., m}
    edge_lengths = []
    for a, b in graph.edges():
        length = model.NewIntVar(1, m, f"len_{a}_{b}")
        model.AddAbsEquality(length, labels[a] - labels[b])
        edge_lengths.append(length)
    
    model.AddAllDifferent(edge_lengths)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        labeled_graph = graph.copy()
        nx.set_node_attributes(labeled_graph, {v: solver.Value(labels[v]) for v in graph.nodes()}, "label")
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
    model = cp_model.CpModel()

    # Variables: Each node gets a unique label in {0, ..., 3m}
    labels = {v: model.NewIntVar(0, 3*m, f"label_{v}") for v in kG.nodes()}

    # Constraint 1: Unique labels within each component
    for i in range(k):
        component_nodes = [v for v in kG.nodes() if v[1] == i]
        model.AddAllDifferent([labels[v] for v in component_nodes])

    # Constraint 2: Edge lengths must form a bijection with {1, ..., k}
    edge_lengths = []
    
    for (u, v) in kG.edges():
        length = model.NewIntVar(1, k, f"len_{u}_{v}")
        model.AddAbsEquality(length, labels[u] - labels[v])  # |u - v|
        edge_lengths.append(length)

    model.AddAllDifferent(edge_lengths)  # Ensure bijection

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    print(f"Solver Status: {solver.StatusName(status)}")  # Debug output

    if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        labeled_kG = kG.copy()
        nx.set_node_attributes(labeled_kG, {v: solver.Value(labels[v]) for v in kG.nodes()}, "label")
        return labeled_kG
    else:
        print("No valid labeling found. Adjust constraints.")
        return None  # No solution found




