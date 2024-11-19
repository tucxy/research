import networkx as nx
import sys
def gvpath(i):
    if i == 0:
        return 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\Graph Theory'
    if i == 1:
        return 'C:\\Users\\Danny\\Desktop\\Git\\research'
    else: 
        return False
sys.path.append(gvpath(0)) # here is the path with GVIS
from graph_visualization import visualize
from itertools import combinations
import math

# how to merge two graphs:
'''
# Create two graphs
G1 = nx.Graph()
G1.add_edges_from([ ('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([ ('c',  'd'),  ('d',  'e')])

# Merge the graphs
G_merged = nx.compose(G1,  G2)
'''
#how to merge multiple graphs
'''
# Create multiple graphs
G1 = nx.Graph()
G1.add_edges_from([ ('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([ ('c',  'd'),  ('d',  'e')])

G3 = nx.Graph()
G3.add_edges_from([ ('e',  'f'),  ('f',  'g')])

# Merge the graphs
G_merged = nx.compose_all([ G1,  G2,  G3])
'''

def inspect(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    Ginfo = {
        "Nodes": nodes,
        "Edges": edges
    }
    
    return Ginfo

def build(vertices,edges):
    """
    Create a graph using NetworkX from a list of vertices and edges.
    build([ u, v, w, ...],  [ (u,  v),  (w,  v),  ...])
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    return G

def merge(*graphs):
    """
    Merge multiple NetworkX graphs into a single graph.
    """
    G = nx.Graph()
    
    for graph in graphs:
        G.add_nodes_from(graph.nodes())
        G.add_edges_from(graph.edges())
    
    return G

def path(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C) - 1):
        G.add_edge(C[ i],C[ i + 1])
    return G

def cycle(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C)):
        G.add_edge(C[ i],C[ (i + 1) % len(C)])
    return G

def star(hub,neighbors):

    leaves = list(neighbors)
    G = nx.Graph()
    G.add_node(hub)
    for node in leaves:
        G.add_node(node)
        G.add_edge(hub,node)
    return G

def trees(n):
    def is_isomorphic_to_any(graph,graph_list):
        for g in graph_list:
            if nx.is_isomorphic(graph,g):
                return True
        return False

    all_trees = [ ]
    nodes = list(range(n + 1))  # A tree with n edges has n+1 nodes

    for edges in combinations(combinations(nodes,  2),n):
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_tree(G) and not is_isomorphic_to_any(G,all_trees):
            all_trees.append(G)

    return all_trees

templates = {
    #T_{2}
    "T-(2,1)": nx.Graph([(1, 2)]), #1-path
    
    # T_{3}
    "T-(3,1)": nx.Graph([(1, 2), (2, 3)]), #2-path
    
    #T_{4}
    "T-(4,1)": nx.Graph([(1, 2), (2, 3), (3, 4)]), #3-path
    "T-(4,2)": nx.Graph([(1, 2), (2, 3), (2, 4)]),  #3-star
    
    #T_{5}
    "T-(5,1)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5)]), #4-path
    "T-(5,2)": nx.Graph([(1, 2), (2, 3), (3, 4), (2, 5)]),  #3-path a branch
    "T-(5,3)": nx.Graph([(1, 2), (2, 3), (2, 4), (2, 5)]),  #4-star
    
    #T_{6}
    "T-(6,1)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]), #5-path
    "T-(6,2)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6)]),  #4-path with branch on v2
    "T-(6,3)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (3, 6)]),  #4-path with branch in middle
    "T-(6,4)": nx.Graph([(1, 2), (2, 3), (2, 4), (4, 5), (4, 6)]),  #4-path with two branches
    "T-(6,5)": nx.Graph([(1, 2), (2, 3), (3, 4), (2, 5), (2, 6)]),  #4-star with 1 leaf extended (1)
    "T-(6,6)": nx.Graph([(1, 2), (2, 3), (2, 4), (2, 5), (2, 6)]),  #5-star
    #T_{7}
    "T-(7,1)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]), #6-path
    "T-(7,2)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7)]), #5-path with branch on v2
    "T-(7,3)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7)]), #5-path with branch on v3
    "T-(7,4)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (2, 7)]), #4-star with 1 leaf extended (2)
    "T-(7,5)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (3, 7)]), #4-star with 2 leaves extended (1)
    "T-(7,6)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]), #3-star with 3 leaves extended (1)
    "T-(7,7)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (3, 7)]), #4-path with two branches one in middle
    "T-(7,8)": nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (4, 7)]), #4-path with two branches none in middle
    "T-(7,9)": nx.Graph([(1, 2), (2, 3), (3, 5), (3, 7), (2, 4), (2, 6)]), #5-star with two branches on one leaf
    "T-(7,10)": nx.Graph([(1, 2), (2, 3), (3, 4), (2, 5), (2, 6), (2, 7)]), #5-star with one leaf extended
    "T-(7,11)": nx.Graph([(1, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]),  #6-star

}

# Verify that all templates are defined
templates.keys()


# Add vertex labels (v_{i}) to the template graphs
for template_name, graph in templates.items():
    nx.set_node_attributes(graph, {node: f"v_{node}" for node in graph.nodes}, "label")

#print(templates)

def find_isomorphism_and_map(graphs, templates):
    """
    Matches each component of each graph in `graphs` to a template graph and maps vertices.

    Parameters:
    - graphs: List of NetworkX graphs to process.
    - templates: Dictionary of template graphs keyed by their names.

    Returns:
    - A list of lists containing bijection dictionaries for each component of each graph.
    """
    result = []

    for G in graphs:
        components = list(nx.connected_components(G))
        graph_bijections = []

        for component in components:
            subgraph = G.subgraph(component)
            bijection = None

            for template_graph in templates.values():
                GM = nx.isomorphism.GraphMatcher(subgraph, template_graph)
                if GM.is_isomorphic():
                    bijection = GM.mapping
                    break

            if bijection:
                graph_bijections.append(bijection)
            else:
                graph_bijections.append(None)

        result.append(graph_bijections)

    return result

def generate_disjoint_unions(mapping_result):
    """
    Generate disjoint union of tuples for each graph using the bijection dictionaries.
    
    Parameters:
    - mapping_result: List of bijection dictionaries for each graph.
    
    Returns:
    - List of strings representing the disjoint union of tuples for each graph.
    """
    result = []

    for graph_bijections in mapping_result:
        components_as_tuples = []
        
        for bijection in graph_bijections:
            if bijection is not None:
                # Sort by values in the bijection and create a tuple of keys
                sorted_keys = [key for key, _ in sorted(bijection.items(), key=lambda item: item[1])]
                components_as_tuples.append(f"({','.join(map(str, sorted_keys))})")
        
        # Join all components with '\\sqcup'
        result.append("\\sqcup".join(components_as_tuples))
    
    return result
#notebook
t = 5
inc = 14*(t-1)
#? 7 (mod 14)

#square with a leaf and two paths

G1 = merge(cycle([2,1,3,0]),path([1,4]),path([11,10]),path([22,66]))
G2 = merge(cycle([0,10,6,11]),path([10,5]),path([3,33]),path([4,66]))
G3 = merge(cycle([4,44,3,55]),path([2,44]),path([1,11]),path([5,66]))
G4 = merge(cycle([2,22,1,33]),path([0,22]),path([5,55]),path([6,66]))

D1 = [ G1, G2, G3, G4]
visualize(14*t+7, D1,  '(2221)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\research\\Dalibor project\\7(mod 14)\\tex')