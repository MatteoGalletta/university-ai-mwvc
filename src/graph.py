import math

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

        # Add weights of selected vertices
        for i in range(self.n):
            if vertexCover[i] == 1:
                total_weight += self.weights[i]
        
        # Check if all edges are covered
        uncovered_edges = 0
        for i in range(self.n):
            for neighbor in self.graph[i]:
                if neighbor > i:  # Process each edge only once
                    if vertexCover[i] == 0 and vertexCover[neighbor] == 0:
                        # Neither endpoint is in the cover - apply penalty
                        uncovered_edges += 1

        total_weight += uncovered_edges * penalty

        return total_weight
    
    def isVertexCover(self, vertexCover):
        assert len(vertexCover) == self.n
        
        # Check each edge to ensure at least one endpoint is in the cover
        for i in range(self.n):
            for neighbor in self.graph[i]:
                if neighbor > i:  # Process each edge only once
                    if vertexCover[i] == 0 and vertexCover[neighbor] == 0:
                        # Found an uncovered edge - not a valid vertex cover
                        return False
        
        # All edges are covered
        return True

    def repair_individual(self, individual):
        uncovered = set()
        for i in range(self.n):
            for j in self.graph[i]:
                if j > i and individual[i] == 0 and individual[j] == 0:
                    uncovered.add((i, j))
        
        # Early exit if already a valid cover
        if not uncovered:
            return individual
            
        # Track which edges each vertex covers
        vertex_to_edges = {}
        
        # Pre-compute edge coverage relationships
        for edge in uncovered:
            i, j = edge
            for v in (i, j):
                if individual[v] == 0:  # Only consider vertices not in cover
                    if v not in vertex_to_edges:
                        vertex_to_edges[v] = set()
                    vertex_to_edges[v].add(edge)
        
        # While there are uncovered edges
        while uncovered:
            # Find vertex with best ratio
            best_vertex = None
            best_ratio = float('inf')
            
            for v, edges in vertex_to_edges.items():
                if edges:  # Vertex covers at least one uncovered edge
                    ratio = self.weights[v] / len(edges)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_vertex = v
            
            if best_vertex is None:
                break
                
            # Add best vertex to cover
            individual[best_vertex] = 1
            
            # Get the edges this vertex covers (make a copy to avoid modification during iteration)
            covered_edges = vertex_to_edges[best_vertex].copy()
            
            # Update data structures
            for edge in covered_edges:
                # Remove the edge from uncovered set
                uncovered.remove(edge)
                
                # Remove this edge from all vertices that could cover it
                for v in vertex_to_edges:
                    vertex_to_edges[v].discard(edge)
            
            # Remove the vertex from consideration
            del vertex_to_edges[best_vertex]
        