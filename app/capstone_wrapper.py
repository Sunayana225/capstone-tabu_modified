"""
Wrapper functions for Capstone algorithms to enable comparison with Tabu Search.
Handles both TSP problems and Job Scheduling problems.
"""

import json
import os
import sys
import time
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64


def _load_tsp_problem_coordinates(problem_name: str) -> List[Tuple[float, float]]:
    """Load TSP problem coordinates from TSPLIB format or use built-in Eil51."""
    
    # Built-in Eil51 coordinates as fallback
    eil51_points: List[Tuple[float, float]] = [
        (37.0,52.0), (49.0,49.0), (52.0,64.0), (20.0,26.0), (40.0,30.0),
        (21.0,47.0), (17.0,63.0), (31.0,62.0), (52.0,33.0), (51.0,21.0),
        (42.0,41.0), (31.0,32.0), (5.0,25.0),  (12.0,42.0), (36.0,16.0),
        (52.0,41.0), (27.0,23.0), (17.0,33.0), (13.0,13.0), (57.0,58.0),
        (62.0,42.0), (42.0,57.0), (16.0,57.0), (8.0,52.0),  (7.0,38.0),
        (27.0,68.0), (30.0,48.0), (43.0,67.0), (58.0,48.0), (58.0,27.0),
        (37.0,69.0), (38.0,46.0), (46.0,10.0), (61.0,33.0), (62.0,63.0),
        (63.0,69.0), (32.0,22.0), (45.0,35.0), (59.0,15.0), (5.0,6.0),
        (10.0,17.0), (21.0,10.0), (5.0,64.0),  (30.0,15.0), (39.0,10.0),
        (32.0,39.0), (25.0,32.0), (25.0,55.0), (48.0,28.0), (56.0,37.0),
        (30.0,40.0)
    ]
    
    # If requesting eil51, return built-in coordinates
    if problem_name.lower() in ["eil51", "eil51.tsp"]:
        return eil51_points
    
    # Try to load from TSPLIB format
    try:
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent
        tsp_dir = root_dir / "tabu-search" / "problems" / "tsp"
        
        # Look for the problem file
        problem_file = None
        problem_clean = problem_name.replace(".tsp", "")
        
        # Debug: Print what we're looking for
        print(f"Loading TSP problem: {problem_name} (cleaned: {problem_clean})")
        
        # Check multiple possible locations
        possible_paths = [
            tsp_dir / f"{problem_clean}.tsp",
            tsp_dir / f"{problem_clean}" / f"{problem_clean}.tsp",
            tsp_dir / f"{problem_clean}.tsp" / f"{problem_clean}.tsp",  # Add this pattern
            tsp_dir / problem_name
        ]
        
        for i, path in enumerate(possible_paths):
            print(f"  Checking path {i+1}: {path} - Exists: {path.exists()}, Is file: {path.is_file() if path.exists() else False}")
            if path.exists() and path.is_file():
                problem_file = path
                print(f"  ✓ Found valid file: {path}")
                break
        
        if problem_file is None:
            print(f"Warning: Problem {problem_name} not found, using Eil51 as fallback")
            return eil51_points
        
        # Parse TSPLIB format
        coordinates = []
        with open(problem_file, 'r') as f:
            lines = f.readlines()
        
        in_coord_section = False
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            elif line == "EOF" or line.startswith("EDGE_WEIGHT_SECTION"):
                break
            elif in_coord_section and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Format: node_id x y
                        x, y = float(parts[1]), float(parts[2])
                        coordinates.append((x, y))
                    except (ValueError, IndexError):
                        continue
        
        if coordinates:
            num_cities = len(coordinates)
            print(f"Loaded {num_cities} coordinates from {problem_name}")
            
            # Warn about large problems
            if num_cities > 1000:
                print(f"⚠️  WARNING: {problem_name} is a large TSP problem with {num_cities} cities.")
                print(f"   This may take a very long time to solve. Consider:")
                print(f"   - Using fewer iterations (50-200 instead of default)")
                print(f"   - Reducing population size for GA (20-50 instead of 100)")
                print(f"   - Using smaller tabu list sizes")
                print(f"   - Or try a smaller problem like berlin52, att48, or eil51 first")
            
            return coordinates
        else:
            print(f"Warning: No coordinates found in {problem_name}, using Eil51 as fallback")
            return eil51_points
            
    except Exception as e:
        print(f"Error loading {problem_name}: {e}, using Eil51 as fallback")
        return eil51_points


def _get_available_tsp_problems() -> List[str]:
    """Get list of available TSP problems."""
    try:
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent
        tsp_dir = root_dir / "tabu-search" / "problems" / "tsp"
        
        problems = []
        if tsp_dir.exists():
            for item in tsp_dir.iterdir():
                if item.is_dir():
                    # Check if this is a TSP problem directory (ends with .tsp)
                    if item.name.endswith('.tsp'):
                        # Look for a file with the same name as the directory
                        tsp_file = item / item.name
                        if tsp_file.exists():
                            # The problem name is the directory name without .tsp extension
                            problem_name = item.name.replace('.tsp', '')
                            problems.append(problem_name)
        
        # Always include eil51 as it's built-in
        if "eil51" not in problems:
            problems.append("eil51")
        
        return sorted(problems)  # Return all problems without limit
    except Exception as e:
        print(f"Error getting TSP problems: {e}")
        return ["eil51"]  # Fallback


def _get_available_d2d_problems() -> List[str]:
    """Get list of available D2D (Door-to-Door) problems."""
    try:
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent
        d2d_dir = root_dir / "tabu-search" / "problems" / "d2d" / "random_data"
        
        problems = []
        if d2d_dir.exists():
            for item in d2d_dir.iterdir():
                if item.is_file() and item.name.endswith('.txt'):
                    problems.append(item.name.replace('.txt', ''))
        
        return sorted(problems)[:100]  # Limit for UI performance
    except Exception:
        return []  # No fallback for d2d


def _get_all_available_problems() -> Dict[str, List[str]]:
    """Get all available problems categorized by type."""
    return {
        "tsp": _get_available_tsp_problems(),
        "d2d": _get_available_d2d_problems()
    }


def get_problem_recommendations() -> Dict[str, List[str]]:
    """Get TSP problem recommendations by size category."""
    return {
        "small": ["gr17", "fri26", "bayg29", "bays29", "dantzig42", "swiss42"],  # < 50 cities
        "medium": ["att48", "gr48", "eil51", "berlin52", "st70", "eil76", "pr76"],  # 50-100 cities  
        "large": ["kroA100", "kroB100", "kroC100", "kroD100", "kroE100", "eil101", "lin105", "ch130"],  # 100-200 cities
        "very_large": ["a280", "lin318", "pr439"],  # 200-500 cities
        "huge": ["att532", "rat575", "p654", "gr666", "d657"],  # 500-1000 cities
        "massive": ["pr1002", "si1032", "u1060", "vm1084", "pcb1173"]  # 1000+ cities (not recommended for testing)
    }


def _ensure_capstone_package_loaded() -> Path:
    """Add Capstone directory to sys.path if not already present."""
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    capstone_dir = root_dir / "Capstone"
    if str(capstone_dir) not in sys.path:
        sys.path.insert(0, str(capstone_dir))
    return capstone_dir


def _ensure_tabu_package_loaded() -> Path:
    """Add Capstone directory to sys.path if not already present."""
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    capstone_dir = root_dir / "Capstone"
    if str(capstone_dir) not in sys.path:
        sys.path.insert(0, str(capstone_dir))
    return capstone_dir


def _plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for web display."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64


# ===========================
# TSP ALGORITHM WRAPPERS
# ===========================

def run_tsp_genetic_algorithm(
    problem: str,
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.02,
    runs: int = 1,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run Genetic Algorithm on TSP problem."""
    
    # Load coordinates first to check problem size
    coordinates = _load_tsp_problem_coordinates(problem)
    num_cities = len(coordinates)
    
    # Adjust parameters for large problems - more aggressive for very large problems
    if num_cities > 1500:
        # Extremely large problems - very aggressive reduction
        original_pop = population_size
        original_gen = generations
        population_size = min(population_size, 30)  # Very small population
        generations = min(generations, 50)  # Very few generations
        print(f"⚠️  MASSIVE problem detected ({num_cities} cities)! Using minimal parameters:")
        print(f"   Population: {original_pop} → {population_size}")
        print(f"   Generations: {original_gen} → {generations}")
        print(f"   ⏱️  Estimated time: 5-15 minutes (will run until completion)")
    elif num_cities > 1000:
        original_pop = population_size
        original_gen = generations
        population_size = min(population_size, 50)  # Reduce population for large problems
        generations = min(generations, 100)  # Reduce generations for large problems
        print(f"📊 Auto-adjusting parameters for large problem ({num_cities} cities):")
        print(f"   Population: {original_pop} → {population_size}")
        print(f"   Generations: {original_gen} → {generations}")
        print(f"   ⏱️  Estimated time: 3-8 minutes (will run until completion)")
    
    capstone_dir = _ensure_capstone_package_loaded()
    previous_cwd = Path.cwd()
    os.chdir(capstone_dir)
    
    try:
        # Import the genetic algorithm module
        import importlib.util
        spec = importlib.util.spec_from_file_location("genetic_algo", capstone_dir / "genetic-algo.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load genetic algorithm module")
        ga_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ga_module)
        
        # Use already loaded coordinates
        points = coordinates
        
        start_time = time.perf_counter()
        
        # Create and run GA with timeout protection
        print(f"🚀 Starting Genetic Algorithm execution...")
        print(f"📊 Parameters: Population={population_size}, Generations={generations}")
        
        # No hard timeout - let algorithm work as long as it's making progress
        
        ga = ga_module.GeneticAlgorithm(
            points, 
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            elitism=True
        )
        
        # Run without hard timeout - let it work as long as it's making progress
        if runs > 1:
            print(f"🔄 Running {runs} iterations...")
            print(f"⏱️  This may take several minutes for large problems. Please be patient...")
            stats = ga.batch_run(runs=runs)
            best_solution = ga.best_solution
            best_cost = stats["best"]
        else:
            print(f"🔄 Running single iteration. Algorithm will show progress as it works...")
            print(f"⏱️  Large problems may take 5-15 minutes. The algorithm is working if you see this message!")
            # Add progress callback if available
            best_solution, best_cost = ga.run()
            stats = {"best": best_cost}
        print(f"✅ Algorithm completed successfully!")
        
        elapsed_time = time.perf_counter() - start_time
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(8, 6))
        tour = best_solution
        x = [points[i][0] for i in tour] + [points[tour[0]][0]]
        y = [points[i][1] for i in tour] + [points[tour[0]][1]]
        ax.plot(x, y, 'co-')
        for order, node in enumerate(tour, start=1):
            xi, yi = points[node]
            ax.text(xi + 0.5, yi + 0.5, str(order), fontsize=8, color='blue')
        ax.set_title("Genetic Algorithm - Best TSP Tour")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plot_image = _plot_to_base64(fig)
        
        return {
            "algorithm": "Genetic Algorithm",
            "problem": f"{problem} TSP ({len(points)} cities)",
            "parameters": {
                "population_size": population_size,
                "generations": generations,
                "mutation_rate": mutation_rate,
                "runs": runs
            },
            "solution": {
                "path": best_solution,
                "cost": float(best_cost)
            },
            "coordinates": {
                "x": [p[0] for p in points],
                "y": [p[1] for p in points]
            },
            "elapsed_ms": elapsed_time * 1000,
            "stats": stats,
            "plot_base64": plot_image
        }
        
    finally:
        os.chdir(previous_cwd)


def run_tsp_ant_colony_optimization(
    problem: str,
    n_ants: int = 30,
    n_iterations: int = 200,
    n_best: int = 5,
    decay: float = 0.95,
    alpha: int = 1,
    beta: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run Ant Colony Optimization on TSP problem."""
    
    # Load coordinates first to check problem size
    points = _load_tsp_problem_coordinates(problem)
    num_cities = len(points)
    
    # Adjust parameters for large problems - more aggressive for very large problems
    if num_cities > 1500:
        # Extremely large problems - very aggressive reduction
        original_ants = n_ants
        original_iter = n_iterations
        n_ants = min(n_ants, 15)  # Very few ants
        n_iterations = min(n_iterations, 30)  # Very few iterations
        print(f"⚠️  MASSIVE problem detected ({num_cities} cities)! Using minimal ACO parameters:")
        print(f"   Ants: {original_ants} → {n_ants}")
        print(f"   Iterations: {original_iter} → {n_iterations}")
        print(f"   ⏱️  Estimated time: 5-15 minutes (will run until completion)")
    elif num_cities > 1000:
        original_ants = n_ants
        original_iter = n_iterations
        n_ants = min(n_ants, 20)  # Reduce ants for large problems
        n_iterations = min(n_iterations, 50)  # Reduce iterations for large problems
        print(f"🐜 Auto-adjusting ACO parameters for large problem ({num_cities} cities):")
        print(f"   Ants: {original_ants} → {n_ants}")
        print(f"   Iterations: {original_iter} → {n_iterations}")
        print(f"   ⏱️  Estimated time: 3-8 minutes (will run until completion)")
    
    capstone_dir = _ensure_capstone_package_loaded()
    previous_cwd = Path.cwd()
    os.chdir(capstone_dir)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ant_colony", capstone_dir / "ant-colony-opt.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load ant colony module")
        aco_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aco_module)
        
        start_time = time.perf_counter()
        
        # Add progress tracking (no hard timeout)
        print(f"🚀 Starting Ant Colony Optimization execution...")
        print(f"🐜 Parameters: Ants={n_ants}, Iterations={n_iterations}")
        
        # No hard timeout - let algorithm work as long as it's making progress
        
        aco = aco_module.AntColony(
            points,
            n_ants=n_ants,
            n_best=n_best,
            n_iterations=n_iterations,
            decay=decay,
            alpha=alpha,
            beta=beta
        )
        
        # Run without hard timeout - let it work as long as it's making progress
        print(f"🔄 Running ACO. Algorithm will work until completion...")
        print(f"⏱️  Large problems may take time. The algorithm is active as long as you see this message!")
        best_route, best_distance = aco.run()
        print(f"✅ ACO completed successfully!")
        
        elapsed_time = time.perf_counter() - start_time
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(8, 6))
        tour = best_route
        x = [points[i][0] for i in tour]
        y = [points[i][1] for i in tour]
        ax.plot(x, y, 'co')
        
        a_scale = float(max(x))/float(100)
        ax.arrow(x[-1], y[-1], (x[0]-x[-1]), (y[0]-y[-1]),
                head_width=a_scale, color='g', length_includes_head=True)
        for i in range(len(x)-1):
            ax.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]),
                    head_width=a_scale, color='g', length_includes_head=True)
        
        for order, node in enumerate(tour, start=1):
            xi, yi = points[node]
            ax.text(xi + 0.5, yi + 0.5, str(order), fontsize=8, color='blue')
        
        ax.set_title("Ant Colony Optimization - Best TSP Tour")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plot_image = _plot_to_base64(fig)
        
        return {
            "algorithm": "Ant Colony Optimization",
            "problem": f"{problem} TSP ({len(points)} cities)",
            "parameters": {
                "n_ants": n_ants,
                "n_iterations": n_iterations,
                "n_best": n_best,
                "decay": decay,
                "alpha": alpha,
                "beta": beta
            },
            "solution": {
                "path": best_route,
                "cost": float(best_distance)
            },
            "coordinates": {
                "x": [p[0] for p in points],
                "y": [p[1] for p in points]
            },
            "elapsed_ms": elapsed_time * 1000,
            "fitness_history": aco.fitness_history,
            "plot_base64": plot_image
        }
        
    finally:
        os.chdir(previous_cwd)


def run_tsp_simulated_annealing(
    problem: str,
    T: float = 100,
    alpha: float = 0.995,
    stopping_iter: int = 10000,
    runs: int = 1,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run Simulated Annealing on TSP problem."""
    
    # Load coordinates first to check problem size
    points = _load_tsp_problem_coordinates(problem)
    num_cities = len(points)
    
    # Adjust parameters for large problems - more aggressive for very large problems
    if num_cities > 1500:
        # Extremely large problems - very aggressive reduction
        original_iter = stopping_iter
        stopping_iter = min(stopping_iter, 1000)  # Very few iterations
        print(f"⚠️  MASSIVE problem detected ({num_cities} cities)! Using minimal SA parameters:")
        print(f"   Stopping iterations: {original_iter} → {stopping_iter}")
        print(f"   ⏱️  Estimated time: 5-15 minutes (will run until completion)")
    elif num_cities > 1000:
        original_iter = stopping_iter
        stopping_iter = min(stopping_iter, 2000)  # Reduce iterations for large problems
        print(f"🌡️  Auto-adjusting SA parameters for large problem ({num_cities} cities):")
        print(f"   Stopping iterations: {original_iter} → {stopping_iter}")
        print(f"   ⏱️  Estimated time: 3-8 minutes (will run until completion)")
    
    capstone_dir = _ensure_capstone_package_loaded()
    previous_cwd = Path.cwd()
    os.chdir(capstone_dir)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sim_anneal", capstone_dir / "simulated-annealing.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load simulated annealing module")
        sa_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa_module)
        
        start_time = time.perf_counter()
        
        # Add progress tracking (no hard timeout)
        print(f"🚀 Starting Simulated Annealing execution...")
        print(f"🌡️  Parameters: T={T}, Alpha={alpha}, Stopping_iter={stopping_iter}")
        
        # No hard timeout - let algorithm work as long as it's making progress
        
        sa = sa_module.SimAnneal(points, T=T, alpha=alpha, stopping_iter=stopping_iter)
        
        # Run without hard timeout - let it work as long as it's making progress
        if runs > 1:
            print(f"🔄 Running SA with {runs} iterations...")
            print(f"⏱️  Multiple runs may take time. Algorithm is working as long as you see this!")
            stats = sa.batch_run(runs=runs)
            best_solution = sa.best_solution
            best_cost = stats["best"]
        else:
            print(f"🔄 Running SA. Algorithm will work until completion...")
            print(f"⏱️  Large problems may take time. The algorithm is active!")
            best_solution, best_cost = sa.anneal()
            stats = {"best": best_cost}
        print(f"✅ Simulated Annealing completed successfully!")
        
        elapsed_time = time.perf_counter() - start_time
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(8, 6))
        tour = best_solution
        x = [points[i][0] for i in tour]
        y = [points[i][1] for i in tour]
        
        ax.plot(x, y, 'co')
        a_scale = float(max(x))/float(100)
        ax.arrow(x[-1], y[-1], (x[0]-x[-1]), (y[0]-y[-1]),
                head_width=a_scale, color='g', length_includes_head=True)
        for i in range(len(x)-1):
            ax.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]),
                    head_width=a_scale, color='g', length_includes_head=True)
        
        for order, node in enumerate(tour, start=1):
            xi, yi = points[node]
            ax.text(xi + 0.5, yi + 0.5, str(order), fontsize=8, color='blue')
        
        ax.set_title("Simulated Annealing - Best TSP Tour")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plot_image = _plot_to_base64(fig)
        
        return {
            "algorithm": "Simulated Annealing",
            "problem": f"{problem} TSP ({len(points)} cities)",
            "parameters": {
                "initial_temperature": T,
                "alpha": alpha,
                "stopping_iterations": stopping_iter,
                "runs": runs
            },
            "solution": {
                "path": best_solution,
                "cost": float(best_cost)
            },
            "coordinates": {
                "x": [p[0] for p in points],
                "y": [p[1] for p in points]
            },
            "elapsed_ms": elapsed_time * 1000,
            "stats": stats,
            "plot_base64": plot_image
        }
        
    finally:
        os.chdir(previous_cwd)


# ===========================
# JOB SCHEDULING ALGORITHM WRAPPERS
# ===========================

def run_job_scheduling_tabu_search(
    job_type: str = "basic",  # "basic" or "duration"
    iterations: int = 50,
    tabu_tenure: int = 5,
    neighbors: int = 20,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run Tabu Search on Job Scheduling problem."""
    
    capstone_dir = _ensure_capstone_package_loaded()
    previous_cwd = Path.cwd()
    os.chdir(capstone_dir)
    
    try:
        if job_type == "duration":
            import importlib.util
            spec = importlib.util.spec_from_file_location("tabu_duration", capstone_dir / "tabu-job-diff-duration.py")
            if spec is None or spec.loader is None:
                raise ImportError("Could not load tabu duration module")
            tabu_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tabu_module)
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("tabu_basic", capstone_dir / "tabu-search-job-scheduling.py")
            if spec is None or spec.loader is None:
                raise ImportError("Could not load tabu basic module")
            tabu_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tabu_module)
        
        jobs = tabu_module.jobs
        
        start_time = time.perf_counter()
        
        if job_type == "duration":
            best_seq, best_profit, schedule = tabu_module.tabu_search(
                jobs, iterations=iterations, tabu_tenure=tabu_tenure, neighbors=neighbors
            )
        else:
            best_seq, best_profit = tabu_module.tabu_search(
                jobs, iterations=iterations, tabu_tenure=tabu_tenure, neighbors=neighbors
            )
            _, schedule = tabu_module.evaluate(best_seq, jobs)
        
        elapsed_time = time.perf_counter() - start_time
        
        # Generate schedule visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        
        if job_type == "duration":
            # For duration-based jobs, schedule contains (job_id, start, end)
            for job_id, start, end in schedule:
                ax.barh(0, end-start, left=start-1, edgecolor="black", color="skyblue")
                ax.text((start+end)/2 - 1, 0, f"J{job_id}", ha="center", va="center", fontsize=9)
        else:
            # For basic jobs, schedule is slots array
            for t in range(1, len(schedule)):
                if schedule[t] is not None:
                    job_id = schedule[t]["id"]
                    profit_val = schedule[t]["profit"]
                    ax.barh(0, 1, left=t-1, edgecolor="black", color="skyblue")
                    ax.text(t-0.5, 0, f"J{job_id}\\nP{profit_val}", 
                            ha="center", va="center", fontsize=9)
        
        ax.set_xlabel("Time Slots")
        ax.set_title(f"Tabu Search Job Schedule (Profit={best_profit})")
        ax.set_yticks([])
        plot_image = _plot_to_base64(fig)
        
        return {
            "algorithm": f"Tabu Search ({'Duration-based' if job_type == 'duration' else 'Basic'})",
            "problem": f"Job Scheduling ({len(jobs)} jobs)",
            "parameters": {
                "iterations": iterations,
                "tabu_tenure": tabu_tenure,
                "neighbors": neighbors,
                "job_type": job_type
            },
            "solution": {
                "sequence": best_seq,
                "profit": float(best_profit),
                "schedule": schedule if job_type == "duration" else [
                    {"slot": i, "job": schedule[i]} for i in range(len(schedule)) if schedule[i] is not None
                ]
            },
            "jobs": jobs,
            "elapsed_ms": elapsed_time * 1000,
            "plot_base64": plot_image
        }
        
    finally:
        os.chdir(previous_cwd)


# ===========================
# D2D ALGORITHM WRAPPERS
# ===========================

def run_d2d_tabu_search(
    problem: str,
    iterations: int = 1000,
    shuffle_after: int = 100,
    tabu_size: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run Tabu Search on D2D (Door-to-Door) problem."""
    
    # Add tabu-search directory to path
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    tabu_dir = root_dir / "tabu-search"
    
    previous_cwd = Path.cwd()
    os.chdir(tabu_dir)
    
    try:
        # Import D2D module with proper path setup
        if str(tabu_dir) not in sys.path:
            sys.path.insert(0, str(tabu_dir))
        
        try:
            from ts.d2d.solutions import D2DPathSolution  # type: ignore
            from ts.d2d.neighborhoods.swap import Swap  # type: ignore
            from ts.d2d.neighborhoods.insert import Insert  # type: ignore
        except ImportError as e:
            raise ImportError(f"Failed to import D2D modules. Make sure the tabu-search package is properly installed. Error: {e}")
        
        start_time = time.perf_counter()
        
        # Import problem
        D2DPathSolution.import_problem(problem)
        
        # Configure tabu lists for D2D neighborhoods
        Swap.reset_tabu(maxlen=tabu_size)
        Insert.reset_tabu(maxlen=tabu_size)
        
        # Run D2D tabu search
        solution = D2DPathSolution.tabu_search(
            iterations_count=iterations,
            use_tqdm=verbose,
            shuffle_after=shuffle_after,
        )
        
        elapsed_time = time.perf_counter() - start_time
        
        # Get problem details
        customers_count = getattr(D2DPathSolution, 'customers_count', 0)
        drones_count = getattr(D2DPathSolution, 'drones_count', 0)
        
        # D2D solutions have multiple objectives (cost components)
        cost_components = solution.cost() if hasattr(solution, 'cost') else [0, 0]
        total_cost = sum(cost_components) if isinstance(cost_components, (list, tuple)) else cost_components
        
        return {
            "algorithm": "Tabu Search (D2D)",
            "problem": f"{problem} D2D ({customers_count} customers, {drones_count} drones)",
            "problem_type": "d2d",
            "parameters": {
                "iterations": iterations,
                "tabu_size": tabu_size,
                "shuffle_after": shuffle_after
            },
            "solution": {
                "path": list(solution.path) if hasattr(solution, 'path') else [],
                "cost": float(total_cost),
                "cost_components": cost_components if isinstance(cost_components, (list, tuple)) else [total_cost]
            },
            "problem_details": {
                "customers": customers_count,
                "drones": drones_count,
                "technicians": getattr(D2DPathSolution, 'technicians_count', 0)
            },
            "elapsed_ms": elapsed_time * 1000
        }
        
    finally:
        os.chdir(previous_cwd)


# ===========================
# COMPARISON FUNCTIONS
# ===========================

def run_tsp_algorithm_comparison(
    algorithms: Optional[List[str]] = None,
    problem: str = "eil51",
    verbose: bool = False,
    include_tabu: bool = True
) -> Dict[str, Any]:
    """Compare multiple TSP algorithms including Tabu Search."""
    
    if algorithms is None:
        algorithms = ["genetic_algorithm", "ant_colony_optimization", "simulated_annealing"]
    
    # Always add tabu search algorithms if requested
    if include_tabu:
        algorithms.extend(["tabu_search_modified", "tabu_search_vanilla"])
    
    results = {}
    
    # Default parameters for each algorithm
    params = {
        "genetic_algorithm": {"population_size": 100, "generations": 200, "runs": 1},
        "ant_colony_optimization": {"n_ants": 30, "n_iterations": 150},
        "simulated_annealing": {"T": 100, "stopping_iter": 5000, "runs": 1},
        "tabu_search_modified": {"iterations": 200, "tabu_size": 10, "shuffle_after": 50},
        "tabu_search_vanilla": {"iterations": 200, "tabu_size": 10, "shuffle_after": 50}
    }
    
    for algorithm in algorithms:
        try:
            if algorithm == "genetic_algorithm":
                result = run_tsp_genetic_algorithm(problem, **params[algorithm], verbose=verbose)
            elif algorithm == "ant_colony_optimization":
                result = run_tsp_ant_colony_optimization(problem, **params[algorithm], verbose=verbose)
            elif algorithm == "simulated_annealing":
                result = run_tsp_simulated_annealing(problem, **params[algorithm], verbose=verbose)
            elif algorithm == "tabu_search_modified":
                # Import and run modified tabu search
                from .solver_wrapper import run_tsp
                tabu_result = run_tsp(problem, verbose=verbose, **params[algorithm])
                
                # Ensure proper formatting for comparison UI
                result = {
                    "algorithm": "Tabu Search (Modified)",
                    "problem": f"{problem} TSP ({tabu_result.get('dimension', 'N/A')} cities)",
                    "parameters": tabu_result["parameters"],
                    "solution": tabu_result["solution"],
                    "coordinates": tabu_result.get("coordinates", {}),
                    "elapsed_ms": tabu_result["elapsed_ms"]
                }
                
            elif algorithm == "tabu_search_vanilla":
                # Import and run vanilla tabu search  
                from .solver_wrapper import run_tsp_vanilla
                tabu_params = {
                    "iterations": params[algorithm]["iterations"], 
                    "tabu_size": params[algorithm]["tabu_size"],
                    "max_no_improvement": 100
                }
                tabu_result = run_tsp_vanilla(problem, verbose=verbose, **tabu_params)
                
                # Ensure proper formatting for comparison UI
                result = {
                    "algorithm": "Tabu Search (Vanilla)",
                    "problem": f"{problem} TSP ({tabu_result.get('dimension', 'N/A')} cities)",
                    "parameters": tabu_result["parameters"],
                    "solution": tabu_result["solution"],
                    "coordinates": tabu_result.get("coordinates", {}),
                    "elapsed_ms": tabu_result["elapsed_ms"]
                }
                
            else:
                continue
                
            results[algorithm] = result
            
        except Exception as e:
            print(f"Error running {algorithm}: {e}")  # Debug output
            results[algorithm] = {"error": str(e)}
    
    # Calculate comparison metrics
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if valid_results:
        best_cost = min(r["solution"]["cost"] for r in valid_results.values())
        fastest_time = min(r["elapsed_ms"] for r in valid_results.values())
        
        # Add comparison metrics
        for name, result in valid_results.items():
            cost = result["solution"]["cost"]
            time_ms = result["elapsed_ms"]
            
            result["comparison"] = {
                "cost_ratio": cost / best_cost,
                "time_ratio": time_ms / fastest_time,
                "gap_to_best": ((cost - best_cost) / best_cost) * 100 if best_cost > 0 else 0
            }
    
    return {
        "comparison_type": "TSP Algorithms",
        "problem": problem,
        "algorithms": results,
        "summary": {
            "total_algorithms": len(algorithms),
            "successful_runs": len(valid_results),
            "best_cost": min(r["solution"]["cost"] for r in valid_results.values()) if valid_results else None,
            "algorithm_ranking": sorted(
                valid_results.keys(), 
                key=lambda x: valid_results[x]["solution"]["cost"]
            ) if valid_results else []
        }
    }


def run_all_algorithms_comparison(verbose: bool = False) -> Dict[str, Any]:
    """Run comprehensive comparison of all available algorithms."""
    
    # TSP Comparison
    tsp_comparison = run_tsp_algorithm_comparison(verbose=verbose)
    
    # Job Scheduling Comparison
    job_basic = run_job_scheduling_tabu_search(job_type="basic", verbose=verbose)
    job_duration = run_job_scheduling_tabu_search(job_type="duration", verbose=verbose)
    
    job_comparison = {
        "comparison_type": "Job Scheduling",
        "algorithms": {
            "tabu_search_basic": job_basic,
            "tabu_search_duration": job_duration
        }
    }
    
    return {
        "timestamp": time.time(),
        "tsp_comparison": tsp_comparison,
        "job_scheduling_comparison": job_comparison,
        "summary": {
            "total_algorithms_tested": 5,  # 3 TSP + 2 Job scheduling
            "tsp_algorithms": len(tsp_comparison["algorithms"]),
            "job_scheduling_algorithms": len(job_comparison["algorithms"])
        }
    }