import random
import networkx as nx
#! important notations: [n] = {1,...,n}

class GeneticAlgorithm:
    """Then initializes self environment, and takes graph as object. Initializes population_size to an integer, generations to an integer and mutation_rate to a float<=1"""
    def __init__(self, graph, population_size=100, generations=500, mutation_rate=0.1):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.m = graph.number_of_edges()
        self.A, self.B = self._get_bipartition()

    def _get_bipartition(self):
        """Gets vertex partite sets: V(G) = A ⊔ B"""
        return nx.bipartite.sets(self.graph)

    def _initialize_population(self):
        """Generates initial population of labelings."""
        population = []
        for _ in range(self.population_size): #For each i ∈ {1,...,population_size}...
            labeling = {v: random.randint(0, 2*self.m - 1) for v in self.graph.nodes()} #Generates a dictionary whose keys are each vertex of G and each key has some random value in [2m-2]={1,..,2m-2}
            population.append(labeling) #adds this dictionary to the list of labelings at index i
        return population # loop is complete returns the list of labelings

    def _fitness(self, labeling):
        """Calculate fitness of a labeling (higher is better)."""
        fitness = 0
    
        #* Constraint 1: All vertex labels must be unique *#
        if len(set(labeling.values())) != len(labeling): #if |f(V(G))|≠|V(G)|; if a vertex label is not unique...
            return -float('inf')  #set the fitness to -∞; negative infinity, not acceptable at all.
        
        #* Constraint 2: For edges ab∈ E(G), a ∈ A, b ∈ B implies f(a) < f(b) *#
        for a, b in self.graph.edges(): #for (a,b)∈ E(G)...
            if a in self.A and labeling[a] >= labeling[b]: #if a∈ A, and a>b ...
                fitness -= 20 #punish
            elif b in self.A and labeling[b] >= labeling[a]: # else if b ∈ A and b>a ...
                fitness -= 20 #punish
            else: #otherwise a∈ A and a<b or b∈ A and b<a
                fitness += 10 #reward
        
        #* Constraint 3: f(a) - f(b) ≠ m for all a ∈ A, b ∈ B *#
        for a in self.A: #for each a∈ A...
            for b in self.B: #and each b∈ B...
                if abs(labeling[a] - labeling[b]) == self.m: #if f(a) - f(b) = m...
                    fitness -= 10 #punish for each such violation
        
        #* Constraint 4: ℓ(E(G))={1,...,m}; edge lengths are in bijection with [m]
        lengths = []

        for a,b in self.graph.edges(): #for (a,b)∈ E(G)
            if a in self.A: #if a∈ A...
                lengths.append(b-a)
                if b-a> self.m or b-a<1: #and if b-a>m or b-a<1; if b-a is an illegal length
                    fitness -= 20 #punish 
                else: #otherwise if its not then 1 ≤ b-a ≤ m and...
                    fitness += 10 #reward
            elif b in self.A: # otherwise if #if b∈ A...
                    lengths.append(a-b)
                    if a-b> self.m or a-b<1: #and if a-b>m or a-b<1; if a-b is an illegal length
                        fitness -= 20 #punish
                    else: ##otherwise if its not then 1 ≤ a-b ≤ m and...
                        fitness += 10 #reward
        if len(set(lengths)) < self.m: #if the set of all lengths in the labeling is smaller than {0,...,m-1}...
            fitness -= 300
        else: #otherwise it is the right size and therefore the right set of lengths so...
            fitness += 30
        
        return fitness #return the resulting integer fitness score

    def _crossover(self, parent1, parent2):
        """Single-point crossover."""
        nodes = list(self.graph.nodes()) #sets nodes=V(G) and imposes the standard order on it most likely. So it will be indexed via [|V(G)|]
        crossover_point = random.randint(1, len(nodes) - 1) #selects a random crossover point from index 1 to |V(G)|. For all indices < crossover_point, labels will be taken from parent1. For all indices ≥ crossover_point are labels from parent2.
        child = {} #initializes the child label dictionary which will later contain a value for each key v∈ V(G)
        for i in range(crossover_point): #for i< crossover_point...
            child[nodes[i]] = parent1[nodes[i]] #set labeling for vertex of index i in nodes list to be from parent1
        for i in range(crossover_point, len(nodes)): #for crossover_point ≤ i ≤ |V(G)|
            child[nodes[i]] = parent2[nodes[i]] #set labeling for vertex of index i in nodes list to be from parent1
        return child #returns the child labeling of G

    def _mutate(self, labeling):
        """Randomly mutates a labeling at one node. Takes (G,labeling) as input."""
        node = random.choice(list(self.graph.nodes())) #set node to be some random vertex in V(G)...
        labeling[node] = random.randint(0, 2*self.m - 1) #label this node with some random integer in \ZZ_{2m} and inject the new label into labeling
        return labeling #returns resulting labeling

    def evolve(self):
        """Run the genetic algorithm. (1) Selects elite members of population (2) Births children from elites (3) mutates them (4) returns best member of the new population"""
        population = self._initialize_population()
        for _ in range(self.generations):
            #imposes a partial order on population via the standard order on their fitness'
            population = sorted(population, key=lambda x: self._fitness(x), reverse=True)
            
            #Selects top 20% as elites
            elites = population[:int(0.2 * self.population_size)]
            
            #Generates new population via crossover and mutation
            new_population = elites.copy() #initializes new_population as just the elites
            while len(new_population) < self.population_size: #while new_population<population_size... or basically until new_population size is that of the original population
            #selects top parents
                parent1 = max(random.sample(population[:100], k=5), key=lambda x: self._fitness(x)) 
                parent2 = max(random.sample(population[:100], k=5), key=lambda x: self._fitness(x))
                child = self._crossover(parent1, parent2) #births child from parents using crossover function from earlier.
                if random.random() < self.mutation_rate: #If a random number k from 0 to 1 is less than mutation rate, then...
                    child = self._mutate(child) #mutate the child
                new_population.append(child) #add child to the population
            
            population = new_population #overwrites old population with the new one
        
        #Returns the best labeling from the new population; max fitness labeling
        best_labeling = max(population, key=lambda x: self._fitness(x))
        return best_labeling

    def get_labeled_graph(self, labeling):
        """Returns the graph with labels."""
        labeled_graph = self.graph.copy()
        nx.set_node_attributes(labeled_graph, labeling, "label")
        return labeled_graph