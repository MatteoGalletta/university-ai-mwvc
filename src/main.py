import random, os, json, numpy as np
from deap import base, creator, tools

EVAL_LIMIT = 2 * 10**4

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class Graph:
    def __init__(self, graph, weights):
        self.graph = graph
        self.weights = weights
        self.n = len(graph)
        self.m = sum(len(neighbors) for neighbors in graph.values()) // 2
        self.d = self.m / (self.n * (self.n - 1) / 2)
        self.__saveGraphFromEdges(graph)
    
    # graph is a dictionary where the key is the vertex and the value is a list of neighbors
    # the function must save the graph in a variable in the opposite manner: the key is a vertex and the value is a set of vertices that are neighbors
    def __saveGraphFromEdges(self, graph):
        self.graph_from_edges = {}

        for i in range(self.n):
            self.graph_from_edges[i] = set()

        for vertex in graph:
            for neighbor in graph[vertex]:
                self.graph_from_edges[vertex].add(neighbor)
    
    '''
    @param vertexCover must be an array of n elements containing 0 or 1

    The function checks if the vertexCover is a valid vertex cover for the graph.
    Tt returns the total weight of the vertices in the vertex cover plus a penalty
    for each uncovered edge.
    '''
    def getVertexCoverFitness(self, vertexCover, penalty = 0):
        total_weight = 0

        assert len(vertexCover) == self.n

        for i in range(self.n):
            if vertexCover[i] == 1:
                total_weight += self.weights[i]
            else:
                for neighbor in self.graph[i]:
                    if vertexCover[neighbor] == 1:
                        break
                else:
                    total_weight += penalty

        return total_weight
    
    def isVertexCover(self, vertexCover):
        for i in range(self.n):
            if vertexCover[i] == 0:
                for neighbor in self.graph[i]:
                    if vertexCover[neighbor] == 1:
                        break
                else:
                    return False
        return True

class MinimumWeightVertexCover:

    def __init__(self, graph):
        self.graph = graph
        self.max_weight = max(graph.weights)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, graph.n)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    
        self.evalCount = 0
        def evalMWVC(individual):
            self.evalCount += 1
            if self.evalCount >= EVAL_LIMIT:
                if self.evalCount == EVAL_LIMIT:
                    print("\tEVAL_LIMIT reached!")
                return self.max_weight * self.graph.n + 1, # guarantee to be the worst
            return graph.getVertexCoverFitness(individual, self.max_weight),

        self.toolbox.register("evaluate", evalMWVC)

        self.toolbox.register("mate", tools.cxOnePoint)

        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def run(self):
       
        pop = self.toolbox.population(n=300)

        CXPB, MUTPB = 0.5, 0.2

        print("\tStart of evolution")

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        #print("  Evaluated %i individuals" % len(pop))

        fits = [ind.fitness.values[0] for ind in pop]

        g = 0

        generations = []
        while self.evalCount < EVAL_LIMIT:
            
            g = g + 1
            #print("-- Generation %i --" % g)

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            #print("  Evaluated %i individuals" % len(invalid_ind))

            pop[:] = offspring

            fits = [ind.fitness.values[0] for ind in pop]

            generations.append({
                "best": min(fits),
                "evals": min(self.evalCount, EVAL_COUNT)
            })
            print("\tbest:", generations[-1]['best'], ", evals:", generations[-1]['evals'])

        best_ind = tools.selBest(pop, 1)[0]
        is_best_correct = self.graph.isVertexCover(best_ind)
        print("\tBest individual is %s, %s (%s)" % (best_ind, best_ind.fitness.values, "CORRECT!" if is_best_correct else "WRONG!"))
        return {
            "generations": generations,
            "best": best_ind,
            "is_best_correct": is_best_correct,
            "objective": self.graph.getVertexCoverFitness(best_ind)
        }


#----------


def read_graphs():
    folder = "../problem/wvcp-instances"

    file_names = os.listdir(folder)

    for file_name in file_names:
        with open(os.path.join(folder, file_name), "r") as file:
            n = int(file.readline())
            weights = list(map(int, file.readline().split()))

            graph = {}
            for i in range(n):
                graph[i] = set()
                for j, value in enumerate(list(map(int, file.readline().split()))):
                    if value == 1:
                        graph[i].add(j)

            yield (file_name, Graph(graph, weights))

def main():
    random.seed(64)

    import shutil
    shutil.rmtree('../results', ignore_errors=True)

    for graph_raw in read_graphs():
        file_name, graph = graph_raw
        #print("Graph: ", file_name, graph.graph, "with weights", graph.weights)
        print("Graph: ", file_name)
        metrics = MinimumWeightVertexCover(graph).run()
        
        output_folder = "../results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        metrics_file = os.path.join(output_folder, f"{file_name}.json")
        with open(metrics_file, "w") as file:
            json.dump(metrics, file)
    
    #MinimumWeightVertexCover({}).run({})

if __name__ == "__main__":
    main()
