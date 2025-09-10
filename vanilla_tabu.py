#!/usr/bin/env python3
"""
Vanilla Tabu Search Implementation for TSP Problems

This module implements the original tabu search algorithm with comprehensive
timing and performance metrics. It provides a clean, straightforward implementation
of the classic tabu search metaheuristic for solving Traveling Salesman Problems.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import random
import itertools
import re
from math import sqrt
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING

import matplotlib.pyplot as plt
from tqdm import tqdm

if TYPE_CHECKING:
    from typing_extensions import Self


class TSPProblem:
    """Represents a TSP problem instance"""
    
    def __init__(self, name: str, dimension: int, coordinates: List[Tuple[float, float]], 
                 distances: List[List[float]]):
        self.name = name
        self.dimension = dimension
        self.coordinates = coordinates
        self.distances = distances


class TSPSolution:
    """Represents a TSP solution with path and cost"""
    
    def __init__(self, path: List[int], problem: TSPProblem):
        self.path = path
        self.problem = problem
        self._cost = None
    
    def cost(self) -> float:
        """Calculate the total cost of the solution"""
        if self._cost is None:
            total_cost = 0.0
            for i in range(len(self.path)):
                current = self.path[i]
                next_city = self.path[(i + 1) % len(self.path)]
                total_cost += self.problem.distances[current][next_city]
            self._cost = total_cost
        return self._cost
    
    def copy(self) -> TSPSolution:
        """Create a copy of this solution"""
        return TSPSolution(self.path.copy(), self.problem)
    
    def __lt__(self, other: TSPSolution) -> bool:
        """Compare solutions by cost (for minimization)"""
        return self.cost() < other.cost()
    
    def __eq__(self, other: TSPSolution) -> bool:
        """Check if solutions are equal"""
        return self.path == other.path


class TabuList:
    """Manages the tabu list for tabu search"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.moves: List[Tuple[int, int]] = []  # List of (i, j) swaps
    
    def add_move(self, move: Tuple[int, int]):
        """Add a move to the tabu list"""
        self.moves.append(move)
        if len(self.moves) > self.max_size:
            self.moves.pop(0)
    
    def is_tabu(self, move: Tuple[int, int]) -> bool:
        """Check if a move is tabu"""
        return move in self.moves
    
    def clear(self):
        """Clear the tabu list"""
        self.moves.clear()


class VanillaTabuSearch:
    """Vanilla Tabu Search implementation for TSP"""
    
    def __init__(self, problem: TSPProblem, tabu_size: int = 10, 
                 max_iterations: int = 1000, max_no_improvement: int = 100):
        self.problem = problem
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        
        # Metrics tracking
        self.metrics = {
            'start_time': 0.0,
            'end_time': 0.0,
            'total_time': 0.0,
            'iterations': 0,
            'improvements': 0,
            'tabu_moves': 0,
            'best_cost': float('inf'),
            'initial_cost': 0.0,
            'cost_history': [],
            'improvement_history': []
        }
    
    def generate_initial_solution(self) -> TSPSolution:
        """Generate initial solution using nearest neighbor heuristic"""
        path = [0]  # Start from city 0
        unvisited = set(range(1, self.problem.dimension))
        
        while unvisited:
            current = path[-1]
            nearest = min(unvisited, key=lambda city: self.problem.distances[current][city])
            path.append(nearest)
            unvisited.remove(nearest)
        
        return TSPSolution(path, self.problem)
    
    def generate_neighbors(self, solution: TSPSolution) -> List[Tuple[TSPSolution, Tuple[int, int]]]:
        """Generate all 2-opt neighbors of the current solution"""
        neighbors = []
        n = len(solution.path)
        
        for i in range(n):
            for j in range(i + 2, n):
                # Create 2-opt move by reversing segment between i and j
                new_path = solution.path.copy()
                new_path[i:j+1] = reversed(new_path[i:j+1])
                neighbor = TSPSolution(new_path, self.problem)
                move = (i, j)
                neighbors.append((neighbor, move))
        
        return neighbors
    
    def find_best_neighbor(self, current: TSPSolution, tabu_list: TabuList) -> Tuple[Optional[TSPSolution], Tuple[int, int]]:
        """Find the best non-tabu neighbor"""
        neighbors = self.generate_neighbors(current)
        best_neighbor = None
        best_move = None
        best_cost = float('inf')
        
        for neighbor, move in neighbors:
            if not tabu_list.is_tabu(move):
                if neighbor.cost() < best_cost:
                    best_neighbor = neighbor
                    best_move = move
                    best_cost = neighbor.cost()
        
        return best_neighbor, best_move
    
    def solve(self, verbose: bool = True) -> Tuple[TSPSolution, Dict[str, Any]]:
        """Run the vanilla tabu search algorithm"""
        self.metrics['start_time'] = time.perf_counter()
        
        # Initialize
        current = self.generate_initial_solution()
        best = current.copy()
        tabu_list = TabuList(self.tabu_size)
        
        self.metrics['initial_cost'] = current.cost()
        self.metrics['best_cost'] = best.cost()
        self.metrics['cost_history'].append(current.cost())
        
        if verbose:
            print(f"Initial solution cost: {current.cost():.2f}")
        
        no_improvement_count = 0
        
        # Main tabu search loop
        iterations = range(self.max_iterations)
        if verbose:
            iterations = tqdm(iterations, desc="Tabu Search", unit="iter")
        
        for iteration in iterations:
            self.metrics['iterations'] = iteration + 1
            
            # Find best neighbor
            best_neighbor, best_move = self.find_best_neighbor(current, tabu_list)
            
            if best_neighbor is None:
                # No non-tabu neighbors found, break
                break
            
            # Update current solution
            current = best_neighbor
            tabu_list.add_move(best_move)
            
            # Check for improvement
            if current < best:
                best = current.copy()
                self.metrics['improvements'] += 1
                self.metrics['best_cost'] = best.cost()
                self.metrics['improvement_history'].append(iteration + 1)
                no_improvement_count = 0
                
                if verbose:
                    print(f"Iteration {iteration + 1}: New best cost = {best.cost():.2f}")
            else:
                no_improvement_count += 1
                self.metrics['tabu_moves'] += 1
            
            # Record cost history
            self.metrics['cost_history'].append(current.cost())
            
            # Early stopping if no improvement for too long
            if no_improvement_count >= self.max_no_improvement:
                if verbose:
                    print(f"Stopping early: no improvement for {self.max_no_improvement} iterations")
                break
            
            # Update progress bar
            if verbose and hasattr(iterations, 'set_description'):
                iterations.set_description(f"Tabu Search (Best: {best.cost():.2f}, Current: {current.cost():.2f})")
        
        self.metrics['end_time'] = time.perf_counter()
        self.metrics['total_time'] = self.metrics['end_time'] - self.metrics['start_time']
        
        return best, self.metrics
    
    def plot_solution(self, solution: TSPSolution, title: str = "TSP Solution"):
        """Plot the TSP solution"""
        x_coords = [self.problem.coordinates[i][0] for i in solution.path]
        y_coords = [self.problem.coordinates[i][1] for i in solution.path]
        
        # Close the tour
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        plt.figure(figsize=(10, 8))
        plt.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7)
        plt.scatter(x_coords[:-1], y_coords[:-1], c='red', s=50, zorder=5)
        
        # Add city labels
        for i, (x, y) in enumerate(self.problem.coordinates):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f"{title} - Cost: {solution.cost():.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_metrics(self):
        """Plot the search metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cost history
        ax1.plot(self.metrics['cost_history'], 'b-', linewidth=1)
        ax1.set_title('Cost History During Search')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.grid(True, alpha=0.3)
        
        # Improvement points
        if self.metrics['improvement_history']:
            improvement_costs = [self.metrics['cost_history'][i-1] for i in self.metrics['improvement_history']]
            ax1.scatter(self.metrics['improvement_history'], improvement_costs, 
                       c='red', s=30, zorder=5, label='Improvements')
            ax1.legend()
        
        # Improvement intervals
        if len(self.metrics['improvement_history']) > 1:
            intervals = [self.metrics['improvement_history'][i] - self.metrics['improvement_history'][i-1] 
                        for i in range(1, len(self.metrics['improvement_history']))]
            ax2.bar(range(len(intervals)), intervals, alpha=0.7)
            ax2.set_title('Intervals Between Improvements')
            ax2.set_xlabel('Improvement Number')
            ax2.set_ylabel('Iterations Between Improvements')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No improvement intervals to display', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Intervals Between Improvements')
        
        plt.tight_layout()
        plt.show()


def load_tsp_problem(problem_name: str) -> TSPProblem:
    """Load a TSP problem from TSPLIB format"""
    problem_name = problem_name.removesuffix(".tsp")
    problem_file = Path("problems/tsp") / f"{problem_name}.tsp" / f"{problem_name}.tsp"
    
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    with open(problem_file, 'r') as f:
        content = f.read()
    
    # Parse dimension
    dimension_match = re.search(r"DIMENSION\s*:\s*(\d+)", content)
    if not dimension_match:
        raise ValueError("Could not find DIMENSION in problem file")
    dimension = int(dimension_match.group(1))
    
    # Parse edge weight type
    edge_weight_match = re.search(r"EDGE_WEIGHT_TYPE\s*:\s*(\w+)", content)
    if not edge_weight_match:
        raise ValueError("Could not find EDGE_WEIGHT_TYPE in problem file")
    edge_weight_type = edge_weight_match.group(1)
    
    # Parse coordinates
    coordinates = []
    coord_pattern = r"^\s*\d+\s+([\d\.\-+e]+\s+[\d\.\-+e]+)\s*?$"
    for match in re.finditer(coord_pattern, content, flags=re.MULTILINE):
        x, y = map(float, match.group(1).split())
        coordinates.append((x, y))
    
    if len(coordinates) != dimension:
        raise ValueError(f"Expected {dimension} coordinates, found {len(coordinates)}")
    
    # Calculate distances based on edge weight type
    distances = [[0.0] * dimension for _ in range(dimension)]
    
    if edge_weight_type == "EUC_2D":
        # Euclidean 2D distance
        for i, j in itertools.combinations(range(dimension), 2):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            dist = sqrt(dx*dx + dy*dy)
            distances[i][j] = distances[j][i] = dist
    elif edge_weight_type == "ATT":
        # Pseudo-Euclidean distance (ATT format)
        for i, j in itertools.combinations(range(dimension), 2):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            rij = sqrt((dx*dx + dy*dy) / 10.0)
            tij = int(rij + 0.5)
            if tij < rij:
                dist = tij + 1
            else:
                dist = tij
            distances[i][j] = distances[j][i] = dist
    else:
        raise ValueError(f"Unsupported edge weight type: {edge_weight_type}")
    
    return TSPProblem(problem_name, dimension, coordinates, distances)


def print_metrics(metrics: Dict[str, Any]):
    """Print formatted metrics"""
    print("\n" + "="*60)
    print("VANILLA TABU SEARCH RESULTS")
    print("="*60)
    print(f"Problem: {metrics.get('problem_name', 'Unknown')}")
    print(f"Total execution time: {metrics['total_time']:.4f} seconds")
    print(f"Total iterations: {metrics['iterations']}")
    print(f"Improvements found: {metrics['improvements']}")
    print(f"Tabu moves made: {metrics['tabu_moves']}")
    print(f"Initial cost: {metrics['initial_cost']:.2f}")
    print(f"Final best cost: {metrics['best_cost']:.2f}")
    print(f"Improvement: {metrics['initial_cost'] - metrics['best_cost']:.2f} ({((metrics['initial_cost'] - metrics['best_cost']) / metrics['initial_cost'] * 100):.2f}%)")
    print(f"Average time per iteration: {metrics['total_time'] / metrics['iterations']:.6f} seconds")
    print(f"Improvement rate: {metrics['improvements'] / metrics['iterations'] * 100:.2f}%")
    
    if metrics['improvement_history']:
        print(f"First improvement at iteration: {metrics['improvement_history'][0]}")
        print(f"Last improvement at iteration: {metrics['improvement_history'][-1]}")
    
    print("="*60)


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Vanilla Tabu Search for TSP Problems")
    parser.add_argument("problem", type=str, help="TSP problem name (e.g., 'a280', 'berlin52')")
    parser.add_argument("-i", "--iterations", default=1000, type=int, 
                       help="Maximum number of iterations (default: 1000)")
    parser.add_argument("-t", "--tabu-size", default=10, type=int,
                       help="Tabu list size (default: 10)")
    parser.add_argument("-n", "--no-improvement", default=100, type=int,
                       help="Maximum iterations without improvement (default: 100)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed progress and plots")
    parser.add_argument("-p", "--plot", action="store_true",
                       help="Show solution plot")
    parser.add_argument("-m", "--metrics-plot", action="store_true",
                       help="Show metrics plot")
    parser.add_argument("-d", "--dump", type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # Load problem
        print(f"Loading TSP problem: {args.problem}")
        problem = load_tsp_problem(args.problem)
        print(f"Problem loaded: {problem.dimension} cities")
        
        # Create and run tabu search
        tabu_search = VanillaTabuSearch(
            problem=problem,
            tabu_size=args.tabu_size,
            max_iterations=args.iterations,
            max_no_improvement=args.no_improvement
        )
        
        print(f"\nStarting Vanilla Tabu Search...")
        print(f"Parameters: iterations={args.iterations}, tabu_size={args.tabu_size}, max_no_improvement={args.no_improvement}")
        
        best_solution, metrics = tabu_search.solve(verbose=args.verbose)
        
        # Add problem name to metrics
        metrics['problem_name'] = problem.name
        
        # Print results
        print_metrics(metrics)
        
        # Show plots if requested
        if args.plot:
            tabu_search.plot_solution(best_solution, f"Vanilla Tabu Search - {problem.name}")
        
        if args.metrics_plot:
            tabu_search.plot_metrics()
        
        # Save results if requested
        if args.dump:
            results = {
                'problem': problem.name,
                'dimension': problem.dimension,
                'parameters': {
                    'iterations': args.iterations,
                    'tabu_size': args.tabu_size,
                    'max_no_improvement': args.no_improvement
                },
                'solution': {
                    'path': best_solution.path,
                    'cost': best_solution.cost()
                },
                'metrics': metrics
            }
            
            with open(args.dump, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {args.dump}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
