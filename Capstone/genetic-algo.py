import math
import random
import matplotlib.pyplot as plt
import time


# =========================
# Visualization
# =========================
def plotTSP(paths, points):
    """
    Visualize the TSP route with numbered nodes along the tour.
    """
    tour = paths[0]
    x = [points[i][0] for i in tour] + [points[tour[0]][0]]
    y = [points[i][1] for i in tour] + [points[tour[0]][1]]

    plt.figure()
    plt.plot(x, y, 'co-')
    for order, node in enumerate(tour, start=1):
        xi, yi = points[node]
        plt.text(xi + 0.5, yi + 0.5, str(order), fontsize=8, color='blue')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("TSP Route (Nodes Numbered in Visiting Order)")
    plt.show()


# =========================
# Genetic Algorithm for TSP
# =========================
class GeneticAlgorithm:
    def __init__(self, coords, population_size=100, generations=500,
                 mutation_rate=0.02, elitism=True):
        self.coords = coords
        self.N = len(coords)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        self.dist_matrix = self._compute_distances()
        self.population = self._init_population()
        self.best_solution = None
        self.best_fitness = float("inf")
        self.fitness_history = []
        self.improvement_iterations = []
        self.total_improvements = 0
        self.start_time = None
        self.end_time = None

    def _compute_distances(self):
        dist = [[0]*self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                dist[i][j] = math.dist(self.coords[i], self.coords[j])
        return dist

    def _init_population(self):
        """Initialize population with random permutations."""
        base = list(range(self.N))
        population = []
        for _ in range(self.population_size):
            route = base[:]
            random.shuffle(route)
            population.append(route)
        return population

    def _route_distance(self, route):
        return sum(self.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) + \
               self.dist_matrix[route[-1]][route[0]]

    def _fitness(self, route):
        distance = self._route_distance(route)
        return 1 / distance

    def _rank_routes(self):
        fitness_results = [(route, self._fitness(route)) for route in self.population]
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)

    def _selection(self, ranked_routes):
        """Roulette wheel selection."""
        routes = [r for r, _ in ranked_routes]
        fitnesses = [f for _, f in ranked_routes]
        total = sum(fitnesses)
        probs = [f / total for f in fitnesses]
        selected = random.choices(routes, weights=probs, k=self.population_size)
        return selected

    def _crossover(self, parent1, parent2):
        """Ordered crossover (OX)."""
        start, end = sorted(random.sample(range(self.N), 2))
        child = [None] * self.N
        child[start:end] = parent1[start:end]
        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child

    def _mutate(self, route):
        """Swap mutation."""
        for swapped in range(self.N):
            if random.random() < self.mutation_rate:
                swap_with = random.randint(0, self.N - 1)
                route[swapped], route[swap_with] = route[swap_with], route[swapped]
        return route

    def _next_generation(self, ranked_routes):
        selection_results = self._selection(ranked_routes)
        children = []

        if self.elitism:
            elite = ranked_routes[0][0][:]
            children.append(elite)

        for i in range(0, len(selection_results) - 1, 2):
            parent1 = selection_results[i]
            parent2 = selection_results[i + 1]
            child = self._crossover(parent1, parent2)
            children.append(child)

        # Ensure correct population size and mutate (skip elite)
        next_gen = []
        for i, child in enumerate(children[:self.population_size]):
            if self.elitism and i == 0:
                next_gen.append(child)
            else:
                next_gen.append(self._mutate(child))
        return next_gen

    def run(self):
        """Run the GA once."""
        self.start_time = time.perf_counter()

        for generation in range(self.generations):
            ranked_routes = self._rank_routes()
            best_route, best_fit = ranked_routes[0]
            best_distance = 1 / best_fit

            if best_distance < self.best_fitness:
                self.best_fitness = best_distance
                self.best_solution = best_route
                self.total_improvements += 1
                self.improvement_iterations.append(generation)

            self.fitness_history.append(self.best_fitness)
            self.population = self._next_generation(ranked_routes)

        self.end_time = time.perf_counter()
        return self.best_solution, self.best_fitness

    def batch_run(self, runs=5):
        """Run GA multiple times to compute statistics."""
        all_best_fits = []
        runtimes = []
        all_iterations = []
        all_improvements = []
        all_improvement_iters = []

        for r in range(runs):
            self.population = self._init_population()
            self.best_solution = None
            self.best_fitness = float("inf")
            self.fitness_history = []
            self.improvement_iterations = []
            self.total_improvements = 0

            start_time = time.perf_counter()
            best_sol, best_fit = self.run()
            end_time = time.perf_counter()

            all_best_fits.append(best_fit)
            runtimes.append(end_time - start_time)
            all_iterations.append(self.generations)
            all_improvements.append(self.total_improvements)
            all_improvement_iters.append(list(self.improvement_iterations))

            print(f"Run {r+1}: Best={best_fit:.2f}, Iterations={self.generations}, Improvements={self.total_improvements}, Time={end_time-start_time:.4f}s")

        best_overall = min(all_best_fits)
        mean_fit = sum(all_best_fits)/len(all_best_fits)
        std_fit = math.sqrt(sum((x-mean_fit)**2 for x in all_best_fits)/len(all_best_fits))
        mean_time = sum(runtimes)/len(runtimes)
        mean_iterations = sum(all_iterations)/len(all_iterations)
        mean_improvements = sum(all_improvements)/len(all_improvements)
        mean_improvement_iter = sum(sum(lst)/len(lst) if lst else 0 for lst in all_improvement_iters)/runs

        # Known optimal for Eil51
        optimal = 426
        gap = (best_overall - optimal)/optimal*100

        print("\n==============================")
        print(f"Best solution overall: {best_overall:.2f}")
        print(f"Mean best solution: {mean_fit:.2f}")
        print(f"Std deviation: {std_fit:.2f}")
        print(f"Mean runtime: {mean_time:.4f}s")
        print(f"Gap to optimal: {gap:.2f}%")
        print(f"Mean iterations: {mean_iterations:.1f}")
        print(f"Mean improvements: {mean_improvements:.1f}")
        print(f"Mean iteration of improvement: {mean_improvement_iter:.1f}")
        print("==============================")

        return {
            "best": best_overall,
            "mean_fit": mean_fit,
            "std_fit": std_fit,
            "mean_time": mean_time,
            "gap": gap,
            "mean_iterations": mean_iterations,
            "mean_improvements": mean_improvements,
            "mean_improvement_iter": mean_improvement_iter,
            "all_best_fits": all_best_fits,
            "runtimes": runtimes,
            "fitness_history": self.fitness_history,
            "improvement_iters": all_improvement_iters
        }

    def visualize_routes(self):
        plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        plt.plot(range(len(self.fitness_history)), self.fitness_history)
        plt.ylabel("Distance (Fitness)")
        plt.xlabel("Generation")
        plt.title("GA Fitness over Generations")
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
# Run GA and compute measures
# =========================
if __name__ == "__main__":
    ga = GeneticAlgorithm(points, population_size=150, generations=300, mutation_rate=0.02, elitism=True)
    stats = ga.batch_run(runs=3)
    ga.visualize_routes()
    ga.plot_learning()
