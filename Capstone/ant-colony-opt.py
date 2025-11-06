import math
import random
import matplotlib.pyplot as plt
import time

# =========================
# Visualization
# =========================
def plotTSP(paths, points, num_iters=1):
    """
    Visualize the TSP route with numbered nodes along the tour.
    """
    tour = paths[0]  # best solution
    x = [points[i][0] for i in tour]
    y = [points[i][1] for i in tour]

    plt.plot(x, y, 'co')  # plot cities

    a_scale = float(max(x))/float(100)

    # Draw the route with arrows
    plt.arrow(x[-1], y[-1], (x[0]-x[-1]), (y[0]-y[-1]),
              head_width=a_scale, color='g', length_includes_head=True)
    for i in range(len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]),
                  head_width=a_scale, color='g', length_includes_head=True)

    # Annotate nodes in tour order
    for order, node in enumerate(tour, start=1):
        xi, yi = points[node]
        plt.text(xi + 0.5, yi + 0.5, str(order), fontsize=8, color='blue')

    plt.xlim(min(x)*0.95, max(x)*1.05)
    plt.ylim(min(y)*0.95, max(y)*1.05)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("TSP Route (Nodes Numbered in Visiting Order)")
    plt.show()


# =========================
# Ant Colony Optimization
# =========================
class AntColony:
    def __init__(self, coords, n_ants=20, n_best=5, n_iterations=100, decay=0.95, alpha=1, beta=5):
        self.coords = coords
        self.N = len(coords)
        self.n_ants = n_ants
        self.n_best = n_best  # number of best ants that deposit pheromone
        self.n_iterations = n_iterations
        self.decay = decay  # pheromone decay factor
        self.alpha = alpha  # influence of pheromone
        self.beta = beta    # influence of distance
        self.dist_matrix = self._compute_distances()
        self.pheromone = [[1.0 for _ in range(self.N)] for _ in range(self.N)]
        self.best_solution = None
        self.best_distance = float('inf')
        self.fitness_history = []

    def _compute_distances(self):
        dist = [[0]*self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                dist[i][j] = math.sqrt((self.coords[i][0]-self.coords[j][0])**2 +
                                       (self.coords[i][1]-self.coords[j][1])**2)
        return dist

    def _route_distance(self, route):
        distance = sum(self.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        distance += self.dist_matrix[route[-1]][route[0]]  # close the loop
        return distance


    def _select_next_city(self, current_city, visited):
        probabilities = []
        for city in range(self.N):
            if city in visited:
                probabilities.append(0)
            else:
                tau = self.pheromone[current_city][city] ** self.alpha
                eta = (1.0 / self.dist_matrix[current_city][city]) ** self.beta
                probabilities.append(tau * eta)
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        next_city = random.choices(range(self.N), weights=probabilities, k=1)[0]
        return next_city

    def _generate_solution(self):
        start = random.randint(0, self.N-1)
        route = [start]
        unvisited = set(range(self.N))
        unvisited.remove(start)
        while unvisited:
            next_city = self._select_next_city(route[-1], route)
            route.append(next_city)
            unvisited.remove(next_city)
        return route


    def _update_pheromone(self, all_routes):
        # decay
        for i in range(self.N):
            for j in range(self.N):
                self.pheromone[i][j] *= self.decay
        # deposit pheromone
        sorted_routes = sorted(all_routes, key=lambda x: x[1])
        for route, distance in sorted_routes[:self.n_best]:
            for i in range(self.N):
                a, b = route[i], route[(i+1) % self.N]
                self.pheromone[a][b] += 1.0 / distance
                self.pheromone[b][a] += 1.0 / distance

    def run(self):
        for iteration in range(self.n_iterations):
            all_routes = []
            for _ in range(self.n_ants):
                route = self._generate_solution()
                distance = self._route_distance(route)
                all_routes.append((route, distance))
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_solution = route
            self._update_pheromone(all_routes)
            self.fitness_history.append(self.best_distance)
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best distance = {self.best_distance:.2f}")
        return self.best_solution, self.best_distance

    def visualize_routes(self):
        plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        plt.plot([i for i in range(len(self.fitness_history))], self.fitness_history)
        plt.ylabel("Distance (Fitness)")
        plt.xlabel("Iteration")
        plt.title("ACO Fitness over Iterations")
        plt.show()


# =========================
# Eil51 dataset (TSPLIB)
# =========================
points = [
    (37,52), (49,49), (52,64), (20,26), (40,30),
    (21,47), (17,63), (31,62), (52,33), (51,21),
    (42,41), (31,32), (5,25),  (12,42), (36,16),
    (52,41), (27,23), (17,33), (13,13), (57,58),
    (62,42), (42,57), (16,57), (8,52),  (7,38),
    (27,68), (30,48), (43,67), (58,48), (58,27),
    (37,69), (38,46), (46,10), (61,33), (62,63),
    (63,69), (32,22), (45,35), (59,15), (5,6),
    (10,17), (21,10), (5,64),  (30,15), (39,10),
    (32,39), (25,32), (25,55), (48,28), (56,37),
    (30,40)
]


# =========================
# Run ACO
# =========================
if __name__ == "__main__":
    aco = AntColony(points, n_ants=30, n_best=5, n_iterations=200, decay=0.95, alpha=1, beta=5)
    best_route, best_distance = aco.run()
    print(f"\nBest distance found: {best_distance:.2f}")
    aco.visualize_routes()
    aco.plot_learning()
