import math
import random
import matplotlib.pyplot as plt
import time

# =========================
# Visualization
# =========================
def plotGraph(points, edges):
    """Plot the given graph connections before solving TSP."""
    plt.figure()
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.scatter(x, y, c='red')

    # draw edges
    for (i, j, w) in edges:
        xi, yi = points[i]
        xj, yj = points[j]
        plt.plot([xi, xj], [yi, yj], 'k--', alpha=0.5)
        plt.text((xi + xj) / 2, (yi + yj) / 2, str(w), fontsize=7, color='green')

    # annotate nodes
    for idx, (xi, yi) in enumerate(points):
        plt.text(xi + 0.5, yi + 0.5, str(idx), fontsize=9, color='blue')

    plt.title("Given Graph Connections")
    plt.show()


def plotTSP(paths, points):
    """Plot the best tour found on top of the graph."""
    tour = paths[0]
    x = [points[i][0] for i in tour] + [points[tour[0]][0]]
    y = [points[i][1] for i in tour] + [points[tour[0]][1]]

    plt.figure()
    plt.plot(x, y, 'co-')  # plot cities and path
    for order, node in enumerate(tour, start=1):
        xi, yi = points[node]
        plt.text(xi + 0.5, yi + 0.5, str(order), fontsize=8, color='blue')

    plt.title("TSP Route Found")
    plt.show()


# =========================
# Simulated Annealing for Sparse Graphs
# =========================
class SimAnneal:
    def __init__(self, coords, dist_matrix, T=100, alpha=0.995, stopping_T=1e-8, stopping_iter=10000):
        self.coords = coords
        self.N = len(coords)
        self.T = T
        self.T_save = T
        self.alpha = alpha
        self.stopping_temperature = stopping_T
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.dist_matrix = dist_matrix
        self.nodes = [i for i in range(self.N)]
        self.best_solution = None
        self.best_fitness = float("inf")
        self.fitness_list = []

    def fitness(self, solution):
        """Compute total path distance. Return inf if invalid edge."""
        total = 0
        for i in range(self.N):
            a, b = solution[i], solution[(i + 1) % self.N]
            d = self.dist_matrix[a][b]
            if d == math.inf:
                return math.inf  # invalid edge
            total += d
        return total

    def valid_solution(self, solution):
        """Check if all edges in the tour are valid."""
        for i in range(self.N):
            a, b = solution[i], solution[(i + 1) % self.N]
            if self.dist_matrix[a][b] == math.inf:
                return False
        return True

    def nearest_neighbor_init(self):
        """Create a valid initial tour using nearest-neighbor heuristic."""
        unvisited = set(self.nodes)
        current = random.choice(self.nodes)
        tour = [current]
        unvisited.remove(current)

        while unvisited:
            next_node = None
            min_dist = float("inf")
            for candidate in unvisited:
                d = self.dist_matrix[current][candidate]
                if d < min_dist:
                    next_node, min_dist = candidate, d
            if next_node is None or min_dist == float("inf"):
                # No valid continuation → fail
                return None
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        if not self.valid_solution(tour):
            return None
        return tour

    def accept(self, candidate):
        """Metropolis acceptance criterion."""
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness == math.inf:
            return  # reject invalid tours
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            prob = math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)
            if random.random() < prob:
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """Run the simulated annealing process."""
        self.cur_solution = self.nearest_neighbor_init()
        if self.cur_solution is None:
            print("No valid initial path found. Graph may be disconnected.")
            return None, math.inf

        self.cur_fitness = self.fitness(self.cur_solution)

        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            i, j = random.sample(range(self.N), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]

            if not self.valid_solution(candidate):
                self.iteration += 1
                continue  # skip invalid swaps

            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)

        if self.best_solution is None:
            print("No valid tour found after annealing.")
            return None, math.inf

        return self.best_solution, self.best_fitness


# =========================
# Example graph (6 nodes, sparse connections)
# =========================
points = [(0, 0), (2, 1), (2, 4), (5, 2), (6, 5), (8, 3)]
edges = [
    (0,1,2.2), (1,2,3.0), (2,4,3.2),
    (0,3,5.0), (1,3,3.2), (3,4,3.0), (4,5,2.5), (3,5,3.2)
]


# Build distance matrix
N = len(points)
dist_matrix = [[math.inf] * N for _ in range(N)]
for i, j, w in edges:
    dist_matrix[i][j] = w
    dist_matrix[j][i] = w  # undirected

# =========================
# Run & Visualize
# =========================
if __name__ == "__main__":
    plotGraph(points, edges)  # show predefined graph

    sa = SimAnneal(points, dist_matrix, T=100, alpha=0.995, stopping_iter=5000)
    best_sol, best_fit = sa.anneal()

    if best_sol is None or best_fit == math.inf:
        print("No valid TSP route could be found. Check your graph connectivity.")
    else:
        print("Best route:", best_sol)
        print("Best distance:", round(best_fit, 3))
        plotTSP([best_sol], points)
