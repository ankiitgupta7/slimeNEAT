# viz.py
import os
import neat
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import matplotlib.colors as mcolors

# Folder to save visualizations
VISUALIZATION_FOLDER = "visualizations_slime"
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

def assign_node_layers(genome, connections, num_inputs):
    """
    Assigns nodes to layers based on connectivity.
    Input nodes are assigned layer 0.
    """
    node_layers = {node: 0 for node in range(-num_inputs, 0)}
    for src, dst, _ in connections:
        if src in node_layers:
            node_layers[dst] = max(node_layers.get(dst, 0), node_layers[src] + 1)
    max_layer = max(node_layers.values(), default=0)
    return node_layers, max_layer

def visualize_genome(genome, config, generation, fitness):
    """
    Draws the network topology of a genome.
    The resulting image shows nodes (input, hidden, output) with labels,
    and edges colored by weight magnitude.
    The title displays the generation number and fitness.
    """
    G = nx.DiGraph()

    # Define node groups
    input_nodes = sorted(range(-config.genome_config.num_inputs, 0))
    output_nodes = sorted(range(config.genome_config.num_outputs))
    hidden_nodes = sorted(set(genome.nodes.keys()) - set(input_nodes) - set(output_nodes))
    all_nodes = input_nodes + hidden_nodes + output_nodes

    # Extract enabled connections with their weights.
    connections = [
        (cg.key[0], cg.key[1], cg.weight)
        for cg in genome.connections.values() if cg.enabled
    ]

    # Assign nodes to layers.
    node_layers, max_layer = assign_node_layers(genome, connections, config.genome_config.num_inputs)
    # Make sure all hidden nodes are assigned.
    for node in hidden_nodes:
        if node not in node_layers:
            node_layers[node] = 1

    # Calculate positions for each node.
    layer_positions = {}
    node_layers_sorted = {layer: [] for layer in range(max_layer + 1)}
    for node, layer in node_layers.items():
        node_layers_sorted[layer].append(node)
    for layer, nodes in node_layers_sorted.items():
        for i, node in enumerate(sorted(nodes)):
            layer_positions[node] = (layer, i - len(nodes) / 2)

    # Setup a colormap for edge weights.
    weight_values = [abs(w) for (_, _, w) in connections]
    if weight_values:
        max_weight = max(weight_values)
        min_weight = min(weight_values)
    else:
        max_weight, min_weight = 1, 0
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    cmap = plt.cm.coolwarm

    # Draw nodes.
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(
        G, layer_positions, nodelist=input_nodes, node_color="lightblue", node_size=600, label="Inputs"
    )
    nx.draw_networkx_nodes(
        G, layer_positions, nodelist=output_nodes, node_color="salmon", node_size=600, label="Outputs"
    )
    nx.draw_networkx_nodes(
        G, layer_positions, nodelist=hidden_nodes, node_color="lightgreen", node_size=600, label="Hidden"
    )

    # Draw edges with colors based on weight.
    for src, dst, weight in connections:
        if src in layer_positions and dst in layer_positions:
            color = cmap(norm(abs(weight)))
            nx.draw_networkx_edges(
                G, layer_positions, edgelist=[(src, dst)], width=2, edge_color=[color]
            )

    # Label nodes.
    labels = {}
    for node in all_nodes:
        if node in input_nodes:
            labels[node] = f"In {abs(node)}"
        elif node in output_nodes:
            labels[node] = f"Out {node}"
        else:
            labels[node] = f"H{node}"
    nx.draw_networkx_labels(G, layer_positions, labels, font_size=10)

    plt.title(f"Generation {generation} | Fitness: {fitness:.2f}", fontsize=14)
    plt.axis("off")
    image_path = os.path.join(VISUALIZATION_FOLDER, f"gen_{generation}.png")
    plt.savefig(image_path)
    plt.close()

class BestGenomeSaver(neat.reporting.BaseReporter):
    """
    NEAT reporter that saves the best genome from each generation as an image.
    It also keeps track of the best fitness over time.
    """
    def __init__(self, config):
        self.config = config
        self.best_fitness_over_time = []
        self.generation = 0

    def post_evaluate(self, config, population, species, best_genome):
        fitness = best_genome.fitness
        self.best_fitness_over_time.append(fitness)
        visualize_genome(best_genome, config, self.generation, fitness)
        self.generation += 1

def create_evolution_gif():
    """
    Creates a GIF from the saved network visualization images.
    """
    images = []
    # Sort images by generation number.
    files = sorted(
        os.listdir(VISUALIZATION_FOLDER),
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    for file in files:
        image_path = os.path.join(VISUALIZATION_FOLDER, file)
        images.append(imageio.imread(image_path))
    gif_path = os.path.join(VISUALIZATION_FOLDER, "evolution.gif")
    imageio.mimsave(gif_path, images, duration=1.0)
    print(f"Saved evolution GIF at {gif_path}")

def plot_fitness(fitness_history):
    """
    Plots the best genome's fitness over generations.
    """
    import matplotlib.pyplot as plt
    generations = list(range(len(fitness_history)))
    plt.figure(figsize=(8, 6))
    plt.plot(generations, fitness_history, marker='o', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Genome Fitness Over Generations")
    plot_path = os.path.join(VISUALIZATION_FOLDER, "fitness_over_time.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved fitness plot at {plot_path}")
