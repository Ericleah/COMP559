import networkx as nx
from itertools import combinations
from collections import defaultdict
import numpy as np
import time

# Helper function to count subgraphs of size k
def count_subgraphs(graph, k):
    """
    Counts the frequency of all connected subgraphs of size k in a labeled graph.
    """
    subgraph_counts = {}
    for nodes in combinations(graph.nodes(), k):
        subgraph = graph.subgraph(nodes)
        if nx.is_connected(subgraph):
            # Create a tuple of sorted edge labels for the subgraph
            edge_labels = tuple(sorted((graph.nodes[n]['label'], graph.nodes[m]['label']) for n, m in subgraph.edges()))
            subgraph_counts[edge_labels] = subgraph_counts.get(edge_labels, 0) + 1
    return subgraph_counts

def graphlet_kernel(graph1, graph2, k):
    """
    Computes the cosine similarity between two graphs using the graphlet kernel.
    """
    counts1 = count_subgraphs(graph1, k)
    counts2 = count_subgraphs(graph2, k)
    # Compute the dot product
    dot_product = sum(counts1.get(key, 0) * counts2.get(key, 0) for key in set(counts1) | set(counts2))
    # Compute the magnitudes
    magnitude1 = np.sqrt(sum(val ** 2 for val in counts1.values()))
    magnitude2 = np.sqrt(sum(val ** 2 for val in counts2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Avoid division by zero
    return dot_product / (magnitude1 * magnitude2)

# Load MOLT-4 dataset (dummy implementation)
def load_molt4_dataset():
    """
    Generates a larger random graph dataset for testing.
    """
    graphs = []
    for _ in range(10):  # 10 graphs
        G = nx.erdos_renyi_graph(n=30, p=0.1)  # Random graph with 30 nodes and 10% edge probability
        # Assign random labels to nodes and edges
        for node in G.nodes():
            G.nodes[node]['label'] = np.random.randint(0, 3)  # Random node labels
        for edge in G.edges():
            G.edges[edge]['label'] = np.random.randint(0, 2)  # Random edge labels
        # Assign a class label to the graph
        G.graph['label'] = np.random.randint(0, 2)
        graphs.append(G)
    return graphs

# Evaluate running time for different values of k
def evaluate_running_time(graphs):
    """
    Evaluates the running time of the graphlet kernel for different values of k.
    """
    for k in [1, 2, 3, 4]:
        start_time = time.time()
        for graph in graphs:
            count_subgraphs(graph, k)
        print(f"k = {k}, Time = {time.time() - start_time:.2f} seconds")

# Compare similarity between graphs in the same and different classes
def compare_similarity(graphs):
    """
    Compares the similarity between graphs in the same class and different classes.
    """
    # Group graphs by class
    class_to_graphs = defaultdict(list)
    for graph in graphs:
        class_to_graphs[graph.graph['label']].append(graph)

    # Compute similarities
    similarities_same_class = []
    similarities_diff_class = []

    # Similarity within the same class
    for class_label, class_graphs in class_to_graphs.items():
        for i in range(len(class_graphs)):
            for j in range(i + 1, len(class_graphs)):
                similarity = graphlet_kernel(class_graphs[i], class_graphs[j], k=2)
                similarities_same_class.append(similarity)

    # Similarity between different classes
    for class1, class2 in combinations(class_to_graphs.keys(), 2):
        for graph1 in class_to_graphs[class1]:
            for graph2 in class_to_graphs[class2]:
                similarity = graphlet_kernel(graph1, graph2, k=2)
                similarities_diff_class.append(similarity)

    # Analyze results
    print(f"Same Class: Mean = {np.mean(similarities_same_class)}, Std = {np.std(similarities_same_class)}")
    print(f"Different Class: Mean = {np.mean(similarities_diff_class)}, Std = {np.std(similarities_diff_class)}")

# Main function
if __name__ == "__main__":
    # Load dataset
    graphs = load_molt4_dataset()

    # Evaluate running time
    print("Running Time Evaluation:")
    evaluate_running_time(graphs)

    # Compare similarity
    print("\nSimilarity Comparison:")
    compare_similarity(graphs)