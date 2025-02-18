import random, os, json, numpy as np
import inspect
from deap import base, creator, tools
from graph import Graph

EVAL_LIMIT = 2 * 10**4

#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMin)

class FitnessMin(base.Fitness):
    def __init__(self, *args):
        self.weights = (-1.0,)
        super(FitnessMin, self).__init__(*args)
class Individual(list):
    def __init__(self, *args):
        self.fitness = FitnessMin()
        super(Individual, self).__init__(*args)

class MinimumWeightVertexCover:

    def __init__(self, graph):
        self.graph = graph
        self.max_weight = max(graph.weights)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, Individual, self.toolbox.attr_bool, graph.n)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    
        self.evalCount = 0
        def evalMWVC(individual):
            self.evalCount += 1
            if self.evalCount >= EVAL_LIMIT:
                #if self.evalCount == EVAL_LIMIT:
                #    print("\tEVAL_LIMIT reached!")
                return self.max_weight * self.graph.n + 1, # guarantee to be the worst
            return graph.getVertexCoverFitness(individual, self.max_weight),

        self.toolbox.register("evaluate", evalMWVC)
        self.toolbox.register("mate", tools.cxOnePoint)
    
    def run(self, params):
       
        pop = self.toolbox.population(n=params.get("POPULATION_SIZE"))

        CXPB = params.get("CXPB")
        MUTPB = params.get("MUTPB")
        K_TOURNAMENT = params.get("K_TOURNAMENT")
        FUN_NUMBER_OF_GENERATIONS = params.get("NUMBER_OF_GENERATIONS")


        #print("\tStart of evolution")

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            ind.is_valid = self.graph.isVertexCover(ind)

        #print("  Evaluated %i individuals" % len(pop))

        fits = [(ind.fitness.values[0], ind.is_valid) for ind in pop]

        g = 0

        generations = []
        while self.evalCount < EVAL_LIMIT:
            
            g = g + 1
            #print("-- Generation %i --" % g)

            offspring = tools.selTournament(pop, len(pop), tournsize=K_TOURNAMENT)
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                if random.random() < MUTPB:
                    tools.mutFlipBit(mutant, indpb=0.05)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)

            if self.evalCount >= EVAL_LIMIT:
                break

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.is_valid = self.graph.isVertexCover(ind)

            #print("  Evaluated %i individuals" % len(invalid_ind))

            pop[:] = offspring

            fits = [(ind.fitness.values[0], ind.is_valid) for ind in pop]
            valid_pop = list(filter(lambda ind: ind.is_valid, pop))
            if len(valid_pop) == 0:
                # no best
                print("ATTENZIONE! LA GENERAZIONE NON HA NESSUN BEST")
                generations.append({
                    "best": None,
                    "objective": None,
                    "avg": np.mean([f[0] for f in fits]),
                    "evals": min(self.evalCount, EVAL_LIMIT)
                })
                continue
            best_ind = min(valid_pop, key=lambda ind: ind.fitness.values[0])

            best_ind_obj = self.graph.getVertexCoverFitness(best_ind)
            assert best_ind.fitness.values[0] == best_ind_obj
            assert best_ind.is_valid

            generations.append({
                "best": best_ind.fitness.values[0],
                "objective": best_ind_obj,
                "avg": np.mean([f[0] for f in fits]),
                "evals": min(self.evalCount, EVAL_LIMIT)
            })
            #print("\tbest:", generations[-1]['best'], ", evals:", generations[-1]['evals'])

            if FUN_NUMBER_OF_GENERATIONS(self.graph, generations) <= g:
                break

        best_ind = tools.selBest(pop, 1)[0]
        is_best_correct = self.graph.isVertexCover(best_ind)
        #print("\tBest individual is %s, %s (%s)" % (best_ind, best_ind.fitness.values, "CORRECT!" if is_best_correct else "WRONG!"))
        return {
            "generations": generations,
            "best": best_ind.fitness.values[0],
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

            file_name_parts = file_name.split("_")
            if len(file_name_parts) == 3:
                for i in range(10):
                    yield (file_name.replace(".txt", "_%02d.txt" % (i+1)), Graph(graph, weights))
            else:
                yield (file_name, Graph(graph, weights))

def get_problem_class(n):
    if n < 100:
        return "SPI"
    if n <= 500:
        return "MPI"
    return "LPI"

def main():
    random.seed(64)

    #import shutil
    #shutil.rmtree('../results', ignore_errors=True)

    def stopping_criteria(graph, generations):
        if 'same_obj_count' not in graph.__dict__:
            graph.same_obj_count = 0
        
        if 'last_obj' not in graph.__dict__:
            graph.last_obj = None
        
        cur_obj = generations[-1]['objective']
        if graph.last_obj == cur_obj:
            graph.same_obj_count += 1
        else:
            graph.same_obj_count = 0

        # print("\tSame obj count: ", graph.same_obj_count)
        # print("\tCurrent obj: ", cur_obj)
        if graph.same_obj_count >= 100:
            return 0
        
        graph.last_obj = cur_obj
        return len(generations) + 1

    TEST_ITERATION = os.environ.get("TEST_ITERATION", 6)
    META_PARAMETERS = {
        "POPULATION_SIZE": int(os.environ.get("POPULATION_SIZE", 150)),
        "CXPB": float(os.environ.get("CXPB", 0.4)),
        "MUTPB": float(os.environ.get("MUTPB", 0.25)),
        "K_TOURNAMENT": int(os.environ.get("K_TOURNAMENT", 2)),
        "NUMBER_OF_GENERATIONS": stopping_criteria
    }

    from joblib import Parallel, delayed
    def process(graph_raw):
        file_name, graph = graph_raw
        
        def get_minimum_weight(graph, weights):
            """
            Computes the optimal (exact) minimum weight set cover.
            
            Each key in `graph` corresponds to a set, where the effective set is defined
            as graph[i] ∪ {i}. The algorithm returns the optimal collection of keys (sets)
            that covers the entire universe (all keys and all elements in the sets) with
            minimum total weight.
            
            Parameters:
                graph (dict): Keys are set indices; values are sets of elements the set covers.
                weights (list): A list of weights corresponding to each set (indexed by the keys).
                
            Returns:
                selected_sets (list): A list of keys (sets) that form the optimal cover.
                total_weight (number): The total weight of the selected sets.
                
            Note:
                This implementation uses an exponential dynamic programming approach with
                bitmasking and is suitable only for small instances.
            """
            # Build the universe of elements to be covered.
            universe = set()
            for key, covered in graph.items():
                universe.add(key)
                universe.update(covered)
            universe = sorted(universe)  # Sort for consistency.
            element_to_index = {elem: idx for idx, elem in enumerate(universe)}
            n = len(universe)
            
            # Precompute the bitmask for each set.
            # (We treat the effective set as graph[i] ∪ {i}).
            set_bitmasks = {}
            for key, covered in graph.items():
                effective_set = set(covered)
                effective_set.add(key)
                mask = 0
                for e in effective_set:
                    mask |= 1 << element_to_index[e]
                set_bitmasks[key] = mask
            
            # full_mask represents all elements in the universe being uncovered.
            full_mask = (1 << n) - 1

            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(remaining):
                """
                Returns a tuple (min_weight, chosen_sets) for covering the 'remaining' uncovered elements.
                'remaining' is represented as a bitmask.
                """
                if remaining == 0:
                    return (0, ())  # No weight needed if nothing is uncovered.
                
                best = (float('inf'), None)
                # Try choosing each set that covers at least one uncovered element.
                for key, mask in set_bitmasks.items():
                    new_remaining = remaining & ~mask
                    if new_remaining == remaining:
                        continue  # This set doesn't cover any new element.
                    candidate_weight, candidate_sets = dp(new_remaining)
                    total_weight = weights[key] + candidate_weight
                    if total_weight < best[0]:
                        best = (total_weight, (key,) + candidate_sets)
                return best

            total_weight, chosen_sets = dp(full_mask)
            return list(chosen_sets), total_weight
        
        print("Starting to process", file_name, "...")
        minimum_weight = None
        if get_problem_class(graph.n) in ["SPI", "MPI"]:
            minimum_weight = get_minimum_weight(graph.graph, graph.weights)[1]
        performanceMetrics = MinimumWeightVertexCover(graph).run(META_PARAMETERS)
        print("Finished to process", file_name, "| min:", minimum_weight, "| got:", performanceMetrics['objective'], "which is", performanceMetrics['is_best_correct'])
        if minimum_weight is not None and minimum_weight > performanceMetrics['objective']:
            print("ERRORE")

        #performanceMetrics = None
        #if get_problem_class(graph.n) == "LPI":
        performanceMetrics = None#MinimumWeightVertexCover(graph).run(META_PARAMETERS)
        
        return {
            "fileName": file_name,
            "graph": {
                "n": graph.n,
                "m": graph.m,
                "d": graph.d
            },
            "problemClass": get_problem_class(graph.n),
            "testInstanceName": file_name.split("_")[2],
            "testInstanceVersion": file_name.split("_")[3],
            "performanceMetrics": performanceMetrics
        }
    
    # READ GRAPHS
    graphs = read_graphs()
    graphs = sorted(graphs, key=lambda x: x[1].n)#[:2]

    # RUN TESTS
    results = Parallel(n_jobs=4)(delayed(process)(g) for g in graphs)

    # WRITE RESULTS
    output_folder = f"../results/{TEST_ITERATION}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for r in results:
        metrics_file = os.path.join(output_folder, f"{r['problemClass']}_{r['fileName']}.json")
        with open(metrics_file, "w") as file:
            json.dump(r, file)
    
    with open(os.path.join(output_folder, "_meta_parameters.json"), "w") as file:
        # serialize the lambda function as string:
        funcString = str(inspect.getsourcelines(META_PARAMETERS["NUMBER_OF_GENERATIONS"])[0])
        funcString = ':'.join(funcString.strip("['\\n']").split(":")[1:]).strip()
        META_PARAMETERS["NUMBER_OF_GENERATIONS"] = funcString
        json.dump(META_PARAMETERS, file)
    
if __name__ == "__main__":
    main()
