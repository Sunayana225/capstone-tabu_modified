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
# Simulated Annealing
# =========================
class SimAnneal:
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []
        self.improvement_iterations = []
        self.total_improvements = 0
        self.start_time = None
        self.end_time = None

    def initial_solution(self):
        cur_node = random.choice(self.nodes)
        solution = [cur_node]
        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node
        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def dist(self, node_0, node_1):
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0])**2 + (coord_0[1] - coord_1[1])**2)

    def fitness(self, solution):
        return sum(self.dist(solution[i % self.N], solution[(i+1) % self.N]) for i in range(self.N))

    def p_accept(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            self.total_improvements += 1
            self.improvement_iterations.append(self.iteration)
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        self.cur_solution, self.cur_fitness = self.initial_solution()
        self.start_time = time.perf_counter()
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N-1)
            i = random.randint(0, self.N-l)
            candidate[i:(i+l)] = reversed(candidate[i:(i+l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)
        self.end_time = time.perf_counter()
        return self.best_solution, self.best_fitness

    def batch_run(self, runs=5):
        """Run SA multiple times to compute statistics"""
        all_best_fits = []
        runtimes = []
        all_iterations = []
        all_improvements = []
        all_improvement_iters = []

        for r in range(runs):
            self.T = self.T_save
            self.iteration = 1
            self.fitness_list = []
            self.improvement_iterations = []
            self.total_improvements = 0

            start_time = time.perf_counter()
            best_sol, best_fit = self.anneal()
            end_time = time.perf_counter()

            all_best_fits.append(best_fit)
            runtimes.append(end_time - start_time)
            all_iterations.append(self.iteration)
            all_improvements.append(self.total_improvements)
            all_improvement_iters.append(list(self.improvement_iterations))

            print(f"Run {r+1}: Best={best_fit:.2f}, Iterations={self.iteration}, Improvements={self.total_improvements}, Time={end_time-start_time:.4f}s")

        best_overall = min(all_best_fits)
        mean_fit = sum(all_best_fits)/len(all_best_fits)
        std_fit = math.sqrt(sum((x-mean_fit)**2 for x in all_best_fits)/len(all_best_fits))
        mean_time = sum(runtimes)/len(runtimes)
        mean_iterations = sum(all_iterations)/len(all_iterations)
        mean_improvements = sum(all_improvements)/len(all_improvements)
        mean_improvement_iter = sum(sum(lst)/len(lst) if lst else 0 for lst in all_improvement_iters)/runs

        # Gap to known optimal for Eil51
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
            "fitness_history": self.fitness_list,
            "improvement_iters": all_improvement_iters
        }

    def visualize_routes(self):
        plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness (Total Distance)")
        plt.xlabel("Iteration")
        plt.title("SA Fitness over Iterations")
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
# Run SA and compute measures
# =========================
if __name__ == "__main__":
    sa = SimAnneal(points, T=100, alpha=0.995, stopping_iter=10000)
    stats = sa.batch_run(runs=3)
    sa.visualize_routes()
    sa.plot_learning()
