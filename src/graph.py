
class Graph:
    def __init__(self, graph, weights):
        self.graph = graph
        self.weights = weights
        self.n = len(graph)
        self.m = sum(len(neighbors) for neighbors in graph.values()) // 2
        self.d = self.m / (self.n * (self.n - 1) / 2)
        self.__saveGraphFromEdges(graph)
    
    def __saveGraphFromEdges(self, graph):
        self.graph_from_edges = {}

        for i in range(self.n):
            self.graph_from_edges[i] = set()

        for vertex in graph:
            for neighbor in graph[vertex]:
                self.graph_from_edges[vertex].add(neighbor)
    
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
