import networkx as nx
import sys
def gvpath(i):
    if i == 0:
        return 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\Graph Theory'
    if i == 1:
        return 'C:\\Users\\Danny\\Desktop\\Git\\research'
    if i == 2:
        return r'C:\Users\baneg\Desktop\git\research'
    else: 
        return False
sys.path.append(gvpath(2)) # here is the path with GVIS
from graph_visualization import visualize
from itertools import combinations
import math

# how to merge two graphs:
'''
# Create two graphs
G1 = nx.Graph()
G1.add_edges_from([('a', 'b'), ('b', 'c')])

G2 = nx.Graph()
G2.add_edges_from([('c', 'd'), ('d', 'e')])

# Merge the graphs
G_merged = nx.compose(G1, G2)
'''
#how to merge multiple graphs
'''
# Create multiple graphs
G1 = nx.Graph()
G1.add_edges_from([('a', 'b'), ('b', 'c')])

G2 = nx.Graph()
G2.add_edges_from([('c', 'd'), ('d', 'e')])

G3 = nx.Graph()
G3.add_edges_from([('e', 'f'), ('f', 'g')])

# Merge the graphs
G_merged = nx.compose_all([G1, G2, G3])
'''

def inspect(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    Ginfo = {
        "Nodes": nodes,
        "Edges": edges
    }
    
    return Ginfo

def build(vertices, edges):
    """
    Create a graph using NetworkX from a list of vertices and edges.
    build([u,v,w,...], [(u, v), (w, v), ...])
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
        G.add_edge(C[i], C[i + 1])
    return G

def cycle(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C)):
        G.add_edge(C[i], C[(i + 1) % len(C)])
    return G

def star(hub, neighbors):

    leaves = list(neighbors)
    G = nx.Graph()
    G.add_node(hub)
    for node in leaves:
        G.add_node(node)
        G.add_edge(hub, node)
    return G

def K(n):
    G = nx.Graph()
    G.add_nodes_from(range(0,n))
    for node in range(0,n):
        for neighbor in range(0,n):
            if node != neighbor:
                G.add_edge(node,neighbor)
    return G

def trees(n):
    def is_isomorphic_to_any(graph, graph_list):
        for g in graph_list:
            if nx.is_isomorphic(graph, g):
                return True
        return False

    all_trees = []
    nodes = list(range(n + 1))  # A tree with n edges has n+1 nodes

    for edges in combinations(combinations(nodes, 2), n):
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_tree(G) and not is_isomorphic_to_any(G, all_trees):
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

def generate_disjoint_unions(mapping_result, designs):
    """
    Generate disjoint union of tuples for each graph using the bijection dictionaries.
    Preserve the order of numbers within each tuple while sorting the tuples.

    Parameters:
    - mapping_result: List of bijection dictionaries for each graph.
    - designs: List of graphs corresponding to mapping_result.

    Returns:
    - List of strings representing the disjoint union of tuples for each graph.
    """
    result = []

    for graph_bijections, design in zip(mapping_result, designs):
        components_with_templates = []
        
        for bijection in graph_bijections:
            if bijection is not None:
                # Create a tuple of component nodes (preserve the order of the nodes)
                component_nodes = tuple(bijection.keys())

                # Identify template match
                matched_templates = [
                    (name, template) for name, template in templates.items()
                    if nx.is_isomorphic(design.subgraph(component_nodes), template)
                ]
                
                if matched_templates:
                    template_name, _ = matched_templates[0]
                    n, m = map(int, template_name.split("T-(")[1].strip(")").split(","))
                    components_with_templates.append((component_nodes, (n, m)))
                else:
                    # No template matched; append as unmatched component
                    components_with_templates.append((component_nodes, (0, float('inf'))))
        
        # Sort components by the specified criteria
        sorted_components = sorted(
            components_with_templates,
            key=lambda x: (-x[1][0], x[1][1])  # By node count descending, then second index ascending
        )

        # Format sorted tuples for LaTeX while preserving order within each tuple
        result.append("\\sqcup".join(
            f"({','.join(map(str, tpl[0]))})" for tpl in sorted_components
        ))

    return result

def generate_names_first_graph_ordered(designs, templates):
    """
    Generate ordered names for the first graph in each design list, representing disjoint unions of trees,
    formatted for LaTeX as bold math expressions.

    Parameters:
    - designs: List of graph lists where each list represents a design.
    - templates: Dictionary of template graphs keyed by their names.

    Returns:
    - A list of strings representing the disjoint union of templates for the first graph in each design.
    """
    def format_template(name, count):
        # Extract the tuple (i, j) from the name, e.g., "T-(7,1)"
        n, m = map(int, name.split("T-(")[1].strip(")").split(","))
        formatted = f"\\mathbf{{T_{{{n}}}^{{{m}}}}}"
        return f"{count}{formatted}" if count > 1 else formatted

    names = []
    for design in designs:
        # Process only the first graph in the list
        G = design[0]
        
        # Identify components and match each to a template
        components = list(nx.connected_components(G))
        template_counts = {}

        for component in components:
            subgraph = G.subgraph(component)
            matched_templates = [
                name for name, template in templates.items()
                if nx.is_isomorphic(subgraph, template)
            ]

            if matched_templates:
                template_name = matched_templates[0]
                template_counts[template_name] = template_counts.get(template_name, 0) + 1
        
        # Sort templates by the number of nodes (descending) and second index (ascending)
        def sort_key(template_name):
            n, m = map(int, template_name.split("T-(")[1].strip(")").split(","))
            return (-n, m)  # Sort by node count descending, then by second value ascending
        
        sorted_templates = sorted(template_counts.items(), key=lambda x: sort_key(x[0]))
        
        # Format the disjoint union string with LaTeX representation
        disjoint_union = " \\sqcup ".join(
            format_template(name, count) for name, count in sorted_templates
        )
        names.append(disjoint_union)
    
    return names

def generate_latex_table(designs, names):
    """
    Generate a LaTeX table for a list of graph designs, scaled to fit the page width.

    Parameters:
    - designs: List of graph lists where each list represents a design.
    - names: List of names corresponding to each design.

    Returns:
    - A string containing the LaTeX table.
    """
    if len(designs) != len(names):
        raise ValueError("The number of designs must match the number of names.")
    
    table_rows = []
    
    for name, design in zip(names, designs):
        # Generate disjoint unions for the design
        mapping_result = find_isomorphism_and_map(design, templates)
        disjoint_unions = generate_disjoint_unions(mapping_result, design)
        
        # Format disjoint unions for LaTeX, wrapping each in $$ for math mode
        tabular_content = " \\\\ \n        ".join([f"${union}$" for union in disjoint_unions])
        table_rows.append(f"    ${name}$ & \\begin{{tabular}}{{@{{}}l@{{}}}} {tabular_content} \\end{{tabular}}")
    
    # Build table content with two columns
    formatted_rows = []
    for i in range(0, len(table_rows), 2):
        if i + 1 < len(table_rows):  # Pair rows for two-column format
            formatted_rows.append(f"{table_rows[i]} & {table_rows[i + 1]} \\\\\n\\hline")
        else:  # Handle odd number of rows
            formatted_rows.append(f"{table_rows[i]} & \\\\ \n\\hline")
    
    # Combine rows into a LaTeX table
    table = "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{|c|c|c|c|}\n\\hline\n"
    table += "Design Name & Graph Decomposition & Design Name & Graph Decomposition \\\\\n\\hline\n"
    table += "\n".join(formatted_rows)
    table += "\n\\end{tabular}%\n}"
    
    return table

def generate_two_column_longtable(designs, names):
    """
    Generate a LaTeX longtable for graph decompositions in two columns.

    Parameters:
    - designs: list of design graph lists
    - names: list of LaTeX-formatted names (strings)

    Returns:
    - LaTeX longtable as a string
    """
    if len(designs) != len(names):
        raise ValueError("The number of designs must match the number of names.")

    rows = []
    for name, design in zip(names, designs):
        mapping_result = find_isomorphism_and_map(design, templates)
        disjoint_unions = generate_disjoint_unions(mapping_result, design)
        formatted_decomps = " \\\\ \n".join(f"${du}$" for du in disjoint_unions)
        rows.append(
            f"${name}$ & \\begin{{tabular}}{{c}}\n{formatted_decomps}\n\\end{{tabular}} \\\\ \n\\hline"
        )

    table = (
        "\\begin{longtable}{|c|c|}\n"
        "\\hline\n"
        "Design Name & Graph Decomposition \\\\\n"
        "\\hline\n"
        "\\endfirsthead\n"
        "\\hline\n"
        "Design Name & Graph Decomposition \\\\\n"
        "\\hline\n"
        "\\endhead\n"
        + "\n".join(rows) +
        "\n\\end{longtable}"
    )

    return table

#notebook

K21 = nx.complete_graph(21,create_using=None)
K22 = nx.complete_graph(22,create_using=None)
#visualize(21,[K21], 'control', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')

#arbirarily taking vertices from Pauline Cain's construction to be Z_21 
x1,x2,x3,x4,x5,x6,x7 = 0,1,2,3,4,5,6
w1,w2,w3,w4,w5,w6,w7 = 7,8,9,10,11,12,13
v1,v2,v3,v4,v5,v6,v7 = 14,15,16,17,18,19,20

#! (61)-11 for K_21
V1 = merge(star(v1,[v2,v3,v4,v5,v6,v7]),path([x3,x1]))
V2 = merge(star(v2,[w7,v3,v4,v5,v6,v7]),path([x7,x1]))
V3 = merge(star(v3,[w2,w6,v4,v5,v6,v7]),path([w3,x4]))
V4 = merge(star(v4,[w2,w3,w5,v5,v6,v7]),path([v3,x1]))
V5 = merge(star(v5,[w2,w3,w5,w7,v6,v7]),path([x2,x1]))
V6 = merge(star(v6,[w2,w4,w5,w6,w7,v7]),path([v2,x1]))
X2 = merge(star(x2,[w2,w3,w4,w5,w6,w7]),path([v5,w1]))
X3 = merge(star(x3,[x2,w3,w4,w5,w6,w7]),path([v1,w1]))
X4 = merge(star(x4,[x1,x3,x7,w5,w6,w7]),path([w1,w2]))
X5 = merge(star(x5,[x1,x3,x4,w5,w6,w7]),path([w2,w3]))
X6 = merge(star(x6,[x1,x3,x4,x5,w6,w7]),path([w3,w4]))
X7 = merge(star(x7,[x2,x3,x5,x6,w6,w7]),path([v2,w1]))
W1 = merge(star(w1,[x2,x3,x4,x5,x6,x7]),path([v1,x1]))
W2 = merge(star(w2,[x4,x5,x6,x7,v1,v7]),path([w6,v2]))
W3 = merge(star(w3,[x5,x6,x7,v1,v2,v7]),path([v3,w1]))
W4 = merge(star(w4,[v2,v3,v7,x5,x6,x7]),path([v5,x1]))
W5 = merge(star(w5,[v2,v3,v7,x1,x6,x7]),path([v4,x2]))
W6 = merge(star(w6,[v1,v4,v5,v7,x1,w5]),path([w2,x3]))
W7 = merge(star(w7,[v3,v4,v7,x1,w5,w6]),path([v6,x2]))
U1 = merge(star(v1,[x2,x3,x4,x5,x6,x7]),path([w1,v7]))
U2 = merge(star(v2,[x2,x3,x4,x5,x6,x7]),path([v6,w1]))
U3 = merge(star(v3,[x2,x3,x4,x5,x6,x7]),path([v4,w1]))
U4 = merge(star(v4,[x1,x3,x4,x5,x6,x7]),path([w5,v1]))
U5 = merge(star(v5,[x2,x3,x4,x5,x6,x7]),path([w4,v1]))
U6 = merge(star(v6,[x1,x3,x4,x5,x6,x7]),path([w7,v1]))
U7 = merge(star(v7,[x1,x3,x4,x5,x6,x7]),path([w4,w5]))
Y1 = merge(star(w1,[w3,w4,w5,w6,w7,x1]),path([x4,x2]))
Y2 = merge(star(w2,[w4,w5,w6,w7,x1,v2]),path([x5,x2]))
Y3 = merge(star(w3,[w5,w6,w7,x1,v3,v6]),path([x6,x2]))
Y4 = merge(star(w4,[w6,w7,x1,x4,v4,v5]),path([v7,x2]))

K21_611_11 = [V1,V2,V3,V4,V5,V6,X2,X3,X4,X5,X6,X7,W1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,Y1,Y2,Y3,Y4]
F_61_11_7mod14 = merge(V1,V2,V3,V4,V5,V6,X2,X3,X4,X5,X6,X7,W1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,Y1,Y2,Y3,Y4)

#visualize(21,[F_61_11_7mod14], 'K21', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')
#! (61)-11 for K_22
#modified forests:

nX2 = merge(star(x2,[w2,w3,w4,w5,w6,w7]),path([math.inf,x7]))
nY1 = merge(star(w1,[w3,w4,w5,w6,w7,x1]),path([math.inf,v7]))
nW1 = merge(star(w1,[x2,x3,x4,x5,x6,x7]),path([math.inf,w7]))

infx = merge(star(math.inf,[x1,x2,x3,x4,x5,x6]),path([v5,w1]))
infv = merge(star(math.inf,[v1,v2,v3,v4,v5,v6]),path([x4,x2]))
infw = merge(star(math.inf,[w1,w2,w3,w4,w5,w6]),path([v1,x1]))

K22_611_11 = [V1,V2,V3,V4,V5,V6,nX2,X3,X4,X5,X6,X7,nW1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,nY1,Y2,Y3,Y4,infx,infv,infw]
F_61_11_8mod14 = merge(V1,V2,V3,V4,V5,V6,nX2,X3,X4,X5,X6,X7,nW1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,nY1,Y2,Y3,Y4,infx,infv,infw)
#visualize(21,[F_61_11_8mod14], 'K22', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')

#& check cumulative list of stars in both decompositions

stars = [V1,V2,V3,V4,V5,V6,nX2,X2,X3,X4,X5,X6,X7,nW1,W1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,nY1,Y1,Y2,Y3,Y4,infx,infv,infw]

'''for star in stars:
    visualize(21,[star], 'whatever', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')'''

#&last check for isomorphism

#print(nx.is_isomorphic(K21, F_61_11_7mod14 , node_match=None, edge_match=None))
#print(nx.is_isomorphic(K22, F_61_11_8mod14 , node_match=None, edge_match=None))

designs = [K21_611_11
]
names = generate_names_first_graph_ordered(designs, templates)
#print(names)

latex_table = generate_latex_table(designs, names)

latex2col = generate_two_column_longtable(designs, names)
print(latex2col)