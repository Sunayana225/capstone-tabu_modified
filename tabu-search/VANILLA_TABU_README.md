# Vanilla Tabu Search Implementation

This directory contains a clean, original implementation of the Tabu Search metaheuristic for solving Traveling Salesman Problems (TSP). The implementation focuses on the core tabu search algorithm with comprehensive timing and performance metrics.

## Files

- `vanilla_tabu.py` - Main implementation of vanilla tabu search
- `demo_vanilla_tabu.py` - Demonstration script showing usage examples
- `VANILLA_TABU_README.md` - This documentation file

## Features

### Core Algorithm
- **Original Tabu Search**: Implements the classic tabu search metaheuristic
- **2-opt Neighborhood**: Uses 2-opt moves for solution exploration
- **Tabu List Management**: Maintains a tabu list to prevent cycling
- **Aspiration Criteria**: Allows tabu moves if they lead to better solutions

### Performance Metrics
- **Execution Time**: Precise timing using `time.perf_counter()`
- **Iteration Tracking**: Counts total iterations and improvements
- **Cost History**: Records cost progression throughout the search
- **Improvement Analysis**: Tracks when improvements occur
- **Performance Statistics**: Calculates rates and averages

### Problem Support
- **TSPLIB Format**: Supports standard TSP problem files
- **Multiple Distance Types**: EUC_2D and ATT edge weight types
- **Flexible Parameters**: Configurable iterations, tabu size, and stopping criteria

## Usage

### Command Line Interface

```bash
python vanilla_tabu.py <problem_name> [options]
```

#### Arguments
- `problem` - TSP problem name (e.g., 'berlin52', 'att48', 'eil51')

#### Options
- `-i, --iterations` - Maximum number of iterations (default: 1000)
- `-t, --tabu-size` - Tabu list size (default: 10)
- `-n, --no-improvement` - Max iterations without improvement (default: 100)
- `-v, --verbose` - Show detailed progress and plots
- `-p, --plot` - Show solution plot
- `-m, --metrics-plot` - Show metrics plot
- `-d, --dump` - Save results to JSON file

### Examples

```bash
# Basic usage
python vanilla_tabu.py berlin52

# With custom parameters and verbose output
python vanilla_tabu.py att48 -i 200 -t 15 -v

# Save results to file
python vanilla_tabu.py eil51 -i 150 -d results.json

# Show plots
python vanilla_tabu.py berlin52 -p -m
```

### Programmatic Usage

```python
from vanilla_tabu import VanillaTabuSearch, load_tsp_problem

# Load a TSP problem
problem = load_tsp_problem("berlin52")

# Create tabu search instance
tabu_search = VanillaTabuSearch(
    problem=problem,
    tabu_size=10,
    max_iterations=1000,
    max_no_improvement=100
)

# Run the search
best_solution, metrics = tabu_search.solve(verbose=True)

# Access results
print(f"Best cost: {best_solution.cost()}")
print(f"Execution time: {metrics['total_time']:.3f} seconds")
print(f"Improvements: {metrics['improvements']}")
```

## Algorithm Details

### Initial Solution
- Uses nearest neighbor heuristic to generate starting solution
- Greedily selects closest unvisited city at each step

### Neighborhood Structure
- **2-opt Moves**: Reverses segments of the tour
- **Complete Neighborhood**: Evaluates all possible 2-opt moves
- **Move Representation**: Each move is represented as (i, j) indices

### Tabu List Management
- **Fixed Size**: Configurable maximum size (default: 10)
- **FIFO Policy**: First-in-first-out replacement
- **Move Storage**: Stores (i, j) pairs representing 2-opt moves

### Search Strategy
- **Best Improvement**: Always selects the best non-tabu neighbor
- **Aspiration**: Allows tabu moves if they improve the best solution
- **Early Stopping**: Stops if no improvement for specified iterations

## Performance Metrics

The implementation tracks comprehensive metrics:

### Timing Metrics
- `total_time`: Total execution time in seconds
- `start_time` / `end_time`: Precise timing boundaries
- `iterations_per_second`: Search speed

### Search Metrics
- `iterations`: Total iterations performed
- `improvements`: Number of improvements found
- `tabu_moves`: Number of tabu moves made
- `improvement_rate`: Percentage of iterations that found improvements

### Solution Metrics
- `initial_cost`: Cost of starting solution
- `best_cost`: Cost of best solution found
- `improvement`: Absolute improvement in cost
- `improvement_pct`: Percentage improvement

### History Tracking
- `cost_history`: Cost at each iteration
- `improvement_history`: Iterations where improvements occurred

## Example Results

### Berlin52 Problem
```
Problem: berlin52
Total execution time: 1.0692 seconds
Total iterations: 96
Improvements found: 12
Initial cost: 8980.92
Final best cost: 7938.77
Improvement: 1042.14 (11.60%)
Average time per iteration: 0.011137 seconds
Improvement rate: 12.50%
```

### ATT48 Problem
```
Problem: att48
Total execution time: 0.3444 seconds
Total iterations: 40
Improvements found: 10
Initial cost: 12861.00
Final best cost: 11118.00
Improvement: 1743.00 (13.55%)
Average time per iteration: 0.008610 seconds
Improvement rate: 25.00%
```

## Comparison with Modified Version

This vanilla implementation differs from the modified version in the main codebase:

### Vanilla Tabu Search
- **Simple 2-opt neighborhood**: Only 2-opt moves
- **Single tabu list**: One tabu list for all moves
- **Basic aspiration**: Simple best-improvement aspiration
- **No parallelization**: Sequential neighborhood evaluation
- **Clear metrics**: Focused on core algorithm performance

### Modified Version
- **Multiple neighborhoods**: Swap, SegmentShift, SegmentReverse
- **Multiple tabu lists**: Separate tabu lists per neighborhood
- **Advanced features**: Shuffling, post-optimization, parallelization
- **Complex metrics**: Multi-objective and advanced tracking

## Requirements

- Python 3.7+
- matplotlib (for plotting)
- tqdm (for progress bars)

## Installation

```bash
pip install matplotlib tqdm
```

## Running the Demo

```bash
python demo_vanilla_tabu.py
```

This will run the vanilla tabu search on multiple TSP problems and display a comprehensive summary of results and performance metrics.
