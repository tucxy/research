import networkx as nx
from graph_visualization import visualize
from itertools import combinations

import os
output_dir = os.path.dirname(os.path.abspath(__file__)) #PATH string this file is contained in

# how to merge two graphs:
'''
# Create two graphs
G1 = nx.Graph()
G1.add_edges_from([('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([('c',  'd'),  ('d',  'e')])

# Merge the graphs
G_merged = nx.compose(G1,  G2)
'''
#how to merge multiple graphs
'''
# Create multiple graphs
G1 = nx.Graph()
G1.add_edges_from([('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([('c',  'd'),  ('d',  'e')])

G3 = nx.Graph()
G3.add_edges_from([('e',  'f'),  ('f',  'g')])

# Merge the graphs
G_merged = nx.compose_all([G1,  G2,  G3])
'''

def inspect(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    Ginfo = {
        "Nodes": nodes, 
        "Edges": edges
    }
    
    return Ginfo

def build(vertices,  edges):
    """
    Create a graph using NetworkX from a list of vertices and edges.
    build([u, v, w, ...],  [ (u,  v),  (w,  v),  ...])
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
        G.add_edge(C[i],  C[i + 1])
    return G

def cycle(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C)):
        G.add_edge(C[i],  C[(i + 1) % len(C)])
    return G

def star(hub,  neighbors):

    leaves = list(neighbors)
    G = nx.Graph()
    G.add_node(hub)
    for node in leaves:
        G.add_node(node)
        G.add_edge(hub,  node)
    return G

def trees(n):
    def is_isomorphic_to_any(graph,  graph_list):
        for g in graph_list:
            if nx.is_isomorphic(graph,  g):
                return True
        return False

    all_trees = [ ]
    nodes = list(range(n + 1))  # A tree with n edges has n+1 nodes

    for edges in combinations(combinations(nodes,  2),  n):
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_tree(G) and not is_isomorphic_to_any(G,  all_trees):
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
#notebook
t = 2
inc = 14*(t-1)
#? 8 (mod 14)

#! (61)

#*(61)-1

G = merge(path([ 0, 6, 1, 5,  2,  9,  7]),  path([ 3, 4]))

F_61_1 = [ G] #defines the decomposition 'object' a list of graph labelings
#visualize(14*t+7, F_61_1,  '(61)-1',  output_dir)


#*(61)-2

G  = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 5, 3]),  path([ 7, 8]))

F_61_2 = [ G]
#visualize(14*t+7, F_61_2,  '(61)-2',  output_dir)

#*(61)-3

G  = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 2, 4]),  path([ 7, 8]))

F_61_3 = [ G]
#visualize(14*t+7, F_61_3,  '(61)-3',  output_dir)

#*(61)-4

G = merge(star(1, [ 5, 6, 7, 4]), path([ 9, 2, 4]),  path([ 11, 10]))

F_61_4 = [ G]
#visualize(14*t+7, F_61_4,  '(61)-4',  output_dir)

#*(61)-5

G = merge(path([ 3, 8, 1, 4, 2]), path([ 5, 1, 7]),  path([ 9, 10]))

F_61_5 = [ G]
#visualize(14*t+7, F_61_5,  '(61)-5',  output_dir)

#*(61)-6

G = merge(star(8, [ 7, 4, 1]), star(6, [ 1, 0, 3]), path([ 9, 11]))

F_61_6 = [ G]
#visualize(14*t+7, F_61_6,  '(61)-6',  output_dir)

#*(61)-7

G = merge(star(1, [ 8, 5, 6, 7]), path([ 3, 6, 4]), path([ 9, 10]))

F_61_7 = [ G]
#visualize(14*t+7, F_61_7,  '(61)-7',  output_dir)

#*(61)-8

G = merge(star(1, [ 5, 6, 7, 8, 4]), path([ 3, 5, 1]), path([ 9, 10]))

F_61_8 = [ G]
#visualize(14*t+7, F_61_8,  '(61)-8',  output_dir)

#*(61)-9

G = merge(path([ 5, 11, 9, 12, 7]), path([ 9, 10, 6]), path([ 1, 8]))

F_61_9 = [ G]
#visualize(14*t+7, F_61_9,  '(61)-9',  output_dir)

#*(61)-10

G = merge(star(8, [ 1, 4, 5]),  path([ 3, 1, 6, 0]), path([ 9, 10]))

F_61_10 = [ G]
#visualize(14*t+7, F_61_10,  '(61)-10',  output_dir)

#*(61)-11

G = merge(star(1, [ 3 ,4, 5, 6, 7, 8]), path([ 10, 9]))

F_61_11 = [ G]
visualize(14*t+7, F_61_11,  '(61)-11',  output_dir)


#! (52)

#*(52)-1

G = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 12, 10, 11]))

F_52_1 = [ G]
#visualize(14*t+7, F_52_1,  '(52)-1',  output_dir)

#*(52)-2

G = merge(path([ 3, 6, 1, 8, 4]), path([ 0, 6]), path([ 10, 9, 11]))

F_52_2 = [ G]
#visualize(14*t+7, F_52_2,  '(52)-2',  output_dir)

#*(52)-3

G = merge(path([ 5, 11, 9, 12, 7]), path([ 9, 10]), path([ 1, 8, 4]))

F_52_3 = [ G]
#visualize(14*t+7, F_52_3,  '(52)-3',  output_dir)

#*(52)-4

G = merge(path([ 3, 8, 1, 7]), path([ 4, 8, 1, 6]),  path([ 10, 9, 11]))

F_52_4 = [ G]
#visualize(14*t+7, F_52_4,  '(52)-4',  output_dir)

#*(52)-5

G = merge(star(1, [ 8, 5, 4, 7]), path([ 8, 3]), path([ 10, 9, 11]))

F_52_5 = [ G]
#visualize(14*t+7, F_52_5,  '(52)-5',  output_dir)

#*(52)-6

G = merge(star(1, [ 4, 5, 6, 7, 8]), path([ 10, 9, 11]))

F_52_6 = [ G]
#visualize(14*t+7, F_52_6,  '(52)-6',  output_dir)

#! (43)

#*(43)-1

G = merge(path([ 0, 6, 1, 5, 2]), path([ 3, 10, 8, 9]))

F_43_1 = [ G]
#visualize(14*t+7, F_43_1,  '(43)-1',  output_dir)

#*(43)-2

G = merge(path([ 5, 8, 1, 7]), path([ 1, 6]),  path([ 0, 4, 2, 3]))

F_43_2 = [ G]
#visualize(14*t+7, F_43_2,  '(43)-2',  output_dir)

#*(43)-3

G = merge(path([ 4, 8, 1, 7]), path([ 1, 6]), star(9, [ 10, 12, 11]))

F_43_3 = [ G]
#visualize(14*t+7, F_43_3,  '(43)-3',  output_dir)

#*(43)-4

G = merge(star(0, [ 6, 5, 4, 3]), path([ 2, 9, 7, 8]))

F_43_4 = [ G]
#visualize(14*t+7, F_43_4,  '(43)-4',  output_dir)

#*(43)-5

G = merge(star(9, [ 12, 11, 10]), path([ 4, 8, 1, 7, 2]))

F_43_5 = [ G]
#visualize(14*t+7, F_43_5,  '(43)-5',  output_dir)

#*(43)-6

G = merge(star(0, [ 6, 5, 4, 3]), star(9, [ 8, 7, 2]))

F_43_6 = [ G]
#visualize(14*t+7, F_43_6,  '(43)-6',  output_dir)

#! (511)

#*(511)-1

G = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 8, 10]), path([ 3, 4]))

F_511_1 = [ G]
#visualize(14*t+7, F_511_1,  '(511)-1',  output_dir)


#*(511)-2

G = merge(path([ 4, 8, 1, 6, 3]), path([ 6, 0]), path([ 5, 7]), path([ 9, 10]))

F_511_2 = [ G]
#visualize(14*t+7, F_511_2,  '(511)-2',  output_dir)

#*(511)-3

G = merge(star(1, [ 4, 5, 8, 7]), path([ 8, 3]), path([ 0, 2]), path([ 9, 10]))

F_511_3 = [ G]
#visualize(14*t+7, F_511_3,  '(511)-3',  output_dir)

#*(511)-4

G = merge(path([ 5, 8, 1, 7]), path([ 4, 8, 1, 6]),  path([ 0, 2]), path([ 9, 10]))

F_511_4 = [ G]
#visualize(14*t+7, F_511_4,  '(511)-4',  output_dir)

#*(511)-5

G = merge(path([ 5, 11, 9, 12, 7]), path([ 9, 10]), path([ 1, 8]), path([ 0, 4]))

F_511_5 = [ G]
#visualize(14*t+7, F_511_5,  '(511)-5',  output_dir)

#*(511)-6

G = merge(star(1, [ 4, 5, 6, 7, 8]), path([ 2, 3]), path([ 9, 11]))

F_511_6 = [ G]
#visualize(14*t+7, F_511_6,  '(511)-6',  output_dir)

#! (421)

#*(421)-1

G = merge(path([ 0, 6, 1, 5, 2]), path([ 8, 10, 9]), path([ 4, 11]))

F_421_1 = [ G]
#visualize(14*t+7, F_421_1,  '(421)-1',  output_dir)

#*(421)-2

G = merge(path([ 5, 8, 1, 7]), path([ 1, 6]), path([ 10, 9, 11]), path([ 0, 4]))

F_421_2 = [ G]
#visualize(14*t+7, F_421_2,  '(421)-2',  output_dir)

#*(421)-3

G = merge(star(0, [ 6, 5, 4, 3]), path([ 1, 8, 7]), path([ 9, 11]))

F_421_3 = [ G]
#visualize(14*t+7, F_421_3,  '(421)-3',  output_dir)

#! (331)

#*(331)-1

G = merge(path([ 0, 6, 1, 5]), path([ 2, 9, 7, 10]), path([ 3, 4]))

F_331_1 = [ G]
#visualize(14*t+7, F_331_1,  '(331)-1',  output_dir)

#*(331)-2

G = merge(star(0, [ 4, 5, 6]), path([ 11, 9, 10, 7]), path([ 1, 8]))

F_331_2 = [ G]
#visualize(14*t+7, F_331_2,  '(331)-2',  output_dir)

#*(331)-3

G = merge(star(0, [ 4, 5, 6]), star(9, [ 10, 11, 12]), path([ 1, 8]))

F_331_3 = [ G]
#visualize(14*t+7, F_331_3,  '(331)-3',  output_dir)

#! (322)

#*(322)-1

G = merge(path([ 0, 6, 1, 5]), path([ 8, 10, 9]), path([ 11, 4, 7]))

F_322_1 = [ G]
#visualize(14*t+7, F_322_1,  '(322)-1',  output_dir)

#*(322)-2

G = merge(star(0, [ 6, 5, 4]), path([ 1, 8, 7]), path([ 11, 9, 12]))

F_322_2 = [ G]
#visualize(14*t+7, F_322_2,  '(322)-2',  output_dir)

#! (3211)

#*(3211)-1

G = merge(path([ 0, 6, 1, 5]), path([ 8, 10, 7]), path([ 4, 11]), path([ 2, 3]))

F_3211_1 = [ G]
#visualize(14*t+7, F_3211_1,  '(3211)-1',  output_dir)

#*(3211)-2

G = merge(star(0, [ 4, 5, 6]), path([ 11, 9, 12]), path([ 2, 3]), path([ 1, 8]))

F_3211_2 = [ G]
#visualize(14*t+7, F_3211_2,  '(3211)-2',  output_dir)

#! (4111)

#*(4111)-1

G = merge(path([ 0, 6, 1, 5, 2]), path([ 3, 10]), path([ 7, 9]), path([ 11, 12]))

F_4111_1 = [ G]
#visualize(14*t+7, F_4111_1,  '(4111)-1',  output_dir)

#*(4111)-2

G = merge(path([ 4, 8, 1, 7]), path([ 1, 6]), path([ 3, 5]), path([ 9, 12]), path([ 10, 11]))

F_4111_2 = [ G]
#visualize(14*t+7, F_4111_2,  '(4111)-2',  output_dir)

#*(4111)-3

G = merge(star(0, [ 6, 5, 4, 3]), path([ 1, 8]), path([ 10, 11]), path([ 7, 9]))

F_4111_3 = [ G]
#visualize(14*t+7, F_4111_3,  '(4111)-3',  output_dir)

#! (2221)

#*(2221)-1

G = merge(path([ 0, 6, 1]), path([ 4, 8, 5]), path([ 2, 9, 7]), path([ 10, 11]))

F_2221_1 = [ G]
#visualize(14*t+7, F_2221_1,  '(2221)-1',  output_dir)


#! (31111)

#*(31111)-1

G = merge(path([ 0, 6, 1, 5]), path([ 2, 9]), path([ 8, 10]), path([ 4, 7]), path([ 11, 12]))

F_31111_1 = [ G]
#visualize(14*t+7, F_31111_1,  '(31111)-1',  output_dir)

#*(31111)-2

G = merge(star(0, [ 6, 5, 4]), path([ 2, 3]), path([ 9, 11]), path([ 1, 8]), path([ 7, 10]))

F_31111_2 = [ G]
#visualize(14*t+7, F_31111_2,  '(31111)-2',  output_dir)


#! (22111)

#*(22111)-1

G = merge(path([ 0, 6, 1]), path([ 4, 8, 5]), path([ 3, 10]), path([ 7, 9]), path([ 11, 12]))

F_22111_1 = [ G]
#visualize(14*t+7, F_22111_1,  '(22111)-1',  output_dir)

#! (211111)

#*(211111)-1

G = merge(path([ 0, 6, 1]), path([ 4, 8]), path([ 2, 5]), path([ 3, 10]), path([ 7, 9]), path([ 11, 12]))

F_211111_1 = [ G]
#visualize(14*t+7, F_211111_1,  '(211111)-1',  output_dir)
#& above here is generalized

designs = [
    F_61_1, F_61_2, F_61_3, F_61_4, F_61_5, F_61_6, F_61_7, F_61_8, F_61_9, F_61_10,
    F_52_1, F_52_2, F_52_3, F_52_4, F_52_5, F_52_6, F_43_1, F_43_2, F_43_3, F_43_4, 
    F_43_5, F_43_6, F_511_1, F_511_2, F_511_3, F_511_4, F_511_5, F_511_6, F_421_1, 
    F_421_2, F_421_3, F_331_1, F_331_2, F_331_3, F_322_1, F_322_2, F_3211_1, F_3211_2, 
    F_4111_1, F_4111_2, F_4111_3, F_2221_1, F_31111_1, F_31111_2, F_22111_1, F_211111_1
]
names = generate_names_first_graph_ordered(designs, templates)
#print(names)

latex_table = generate_latex_table(designs, names)
#print(latex_table)
