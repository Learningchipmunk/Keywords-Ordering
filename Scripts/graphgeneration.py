import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def BuildRelationList(kw_pairs, input, classifier):
    return [(kw[0], kw[1], classifier(inp)) for kw, inp in zip(kw_pairs, input)]

def BuildGraphDictFromRelationList(relationList):
    """Converts a list of tuples [(kw1, kw2, label), ...] to a graph dict"""
    graph_dict = {}
    for kw1, kw2, label in relationList:
        if(kw1 == kw2):## No 1 cycles in our graphs 
            continue
        elif(label == 1):
            if kw1 in graph_dict.keys():
                graph_dict[kw1].add(kw2)
            else:
                graph_dict[kw1] = set([kw2])
        elif(label == -1):
            if kw2 in graph_dict.keys():
                graph_dict[kw2].add(kw1)
            else:
                graph_dict[kw2] = set([kw1])
        elif(label == 0):
            if not kw2 in graph_dict.keys():
                graph_dict[kw2] = set()
            if not kw1 in graph_dict.keys():
                graph_dict[kw1] = set()


    return graph_dict

def gbm_classifier(embedding_fun, classifier, kw_pair):
    kw1, kw2 = kw_pair
    return classifier([np.concatenate([embedding_fun(kw1), embedding_fun(kw2)])])[0]

def PlotGraph(nodes, graph, starting_node, num_nodes, title):
    """Plots the input graph.
    starting_node and num_nodes are used to indicate which nodes are to be considered in the subgraph.
    """
    subnodes = list(nodes)[starting_node: num_nodes]

    # Og subgraph
    subgraph = graph.subgraph(subnodes)
    pos = nx.spring_layout(subgraph)

    ## Plot Graph
    nx.draw(subgraph, pos=pos, with_labels = True, node_color = 'b')
    plt.title(title)

    plt.show()

def PlotGraphs(nodes, graph1, graph2, starting_node, num_nodes, title1, title2):
    """Plots the two graphs next to each other.
    starting_node and num_nodes are used to indicate which nodes are to be considered in the subgraph.
    """
    subnodes = list(nodes)[starting_node: num_nodes]

    # subgraph1 subgraph
    subgraph1 = graph1.subgraph(subnodes)
    pos1 = nx.spring_layout(subgraph1)

    # subgraph2 subgrapg
    subgraph2 = graph2.subgraph(subnodes)
    pos2 = nx.spring_layout(subgraph2)

    ## Plot with respect to subgraph2
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    # first graph
    plt.subplot(1, 2, 1)
    nx.draw(subgraph1, pos=pos1, with_labels = True, node_color = 'b')
    plt.title(title1)

    # second graph
    plt.subplot(1, 2, 2)
    nx.draw(subgraph2, pos=pos2, with_labels = True, node_color = 'b')
    plt.title(title2)

    plt.show()

    # Compute edit distance between two graphs:
    edit_dist = nx.graph_edit_distance(subgraph1, subgraph2)
    print("The edit disance between both subgraphs is:", edit_dist)


##### Norms for graphs:
def L1norm(mat1, mat2):
    return np.linalg.norm(mat1 - mat2, ord=1)

def L2norm(mat1, mat2):
    return np.linalg.norm(mat1 - mat2, ord=2)

def frobnorm(mat1, mat2):
    return np.linalg.norm(mat1 - mat2, ord='fro')