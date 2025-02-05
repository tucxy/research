import random
import networkx as nx

class GeneticAlgorithm:
    def __init__(self, graph, population_size=50, generations=200):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations

        # **Penalty Weights**
        self.missing_length_penalty = 100  # Increased priority
        self.extra_length_penalty = 80  # Still important, but secondary
        self.partition_violation_penalty = 20  # Lowered to balance priorities

        self.run_data = []

    def _initialize_population(self):
        """Initialize a population of random labelings."""
        population = []
        for _ in range(self.population_size):
            labeling = {v: random.randint(0, 2 * self.graph.number_of_edges() - 2) for v in self.graph.nodes()}
            population.append(labeling)
        return population

    def _fitness(self, labeling):
        """Fitness function ensuring unique edge lengths and correct partitioning."""
        fitness = 0
        labels = list(labeling.values())

        # **Duplicate Labeling Penalty**
        duplicate_penalty_add = 50
        unique_labels = set(labels)
        duplicate_penalty = (-1* duplicate_penalty_add) * (len(labels) - len(unique_labels))
        fitness += duplicate_penalty

        # **Compute Edge Lengths**
        edge_lengths = set(abs(labeling[a] - labeling[b]) for a, b in self.graph.edges())

        m = self.graph.number_of_edges()
        target_lengths = set(range(1, m + 1))

        missing_lengths = target_lengths - edge_lengths
        extra_lengths = edge_lengths - target_lengths

        # **Partitioning Errors**
        partition_A, partition_B = nx.bipartite.sets(self.graph)
        partition_violations = sum(1 for a in partition_A for b in partition_B if labeling[a] >= labeling[b])

        # **Log Data**
        self.run_data.append({
            "missing_lengths": len(missing_lengths),
            "extra_lengths": len(extra_lengths),
            "partition_violations": partition_violations
        })

        # **Penalty Application (Prioritizing Edge Bijection)**
        fitness -= self.missing_length_penalty * len(missing_lengths)  # Increased priority
        fitness -= self.extra_length_penalty * len(extra_lengths)
        fitness -= self.partition_violation_penalty * partition_violations  # Lowered weight

        return fitness

    def _mutate(self, labeling):
        """Perform mutation while preserving edge length bijection."""
        v = random.choice(list(self.graph.nodes()))
        possible_labels = set(range(0, 2 * self.graph.number_of_edges() - 2))

        # Get current edge lengths in use
        current_edges = {(min(labeling[a], labeling[b]), max(labeling[a], labeling[b])) for a, b in self.graph.edges()}
        used_lengths = {abs(a - b) for a, b in current_edges}

        # **Prioritize fixing missing lengths**
        m = self.graph.number_of_edges()
        target_lengths = set(range(1, m + 1))
        missing_lengths = target_lengths - used_lengths

        valid_labels = set()
        for x in possible_labels:
            if all(abs(x - labeling[n]) not in used_lengths for n in self.graph.neighbors(v)):
                valid_labels.add(x)

        if missing_lengths:
            # Pick a mutation that helps satisfy missing lengths
            valid_labels = {x for x in valid_labels if any(abs(x - labeling[n]) in missing_lengths for n in self.graph.neighbors(v))}

        if valid_labels:
            labeling[v] = random.choice(list(valid_labels))

        return labeling

    def evolve(self):
        """Evolve the population over multiple generations & adjust penalties dynamically."""
        population = self._initialize_population()
        for _ in range(self.generations):
            population = sorted(population, key=self._fitness, reverse=True)
            new_population = population[:10]

            while len(new_population) < self.population_size:
                parent1, parent2 = random.choices(population[:10], k=2)
                child = self._crossover(parent1, parent2)
                if random.random() < 0.1:
                    child = self._mutate(child)
                new_population.append(child)

            population = new_population

            # **Adaptive Tuning Based on Run Data**
            if len(self.run_data) > 10:
                avg_missing = sum(d["missing_lengths"] for d in self.run_data) / len(self.run_data)
                avg_extra = sum(d["extra_lengths"] for d in self.run_data) / len(self.run_data)
                avg_partition_violations = sum(d["partition_violations"] for d in self.run_data) / len(self.run_data)

                if avg_missing > 1:
                    self.missing_length_penalty *= 1.05
                elif avg_missing == 0:
                    self.missing_length_penalty *= 0.95

                if avg_extra > 1:
                    self.extra_length_penalty *= 1.05
                elif avg_extra == 0:
                    self.extra_length_penalty *= 0.95

                if avg_partition_violations > 1:
                    self.partition_violation_penalty *= 1.05
                elif avg_partition_violations == 0:
                    self.partition_violation_penalty *= 0.95

        return population[0]

    def get_labeled_graph(self, labeling):
        """Return the graph with the given labeling."""
        labeled_graph = self.graph.copy()
        nx.set_node_attributes(labeled_graph, labeling, "label")
        return labeled_graph
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents while preserving uniqueness."""
        child = {}
        for v in self.graph.nodes():
            if random.random() < 0.5:
                child[v] = parent1[v]
            else:
                child[v] = parent2[v]
        return child

