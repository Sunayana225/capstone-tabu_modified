# Tabu Search Implementation for TSP Problems

A comprehensive implementation of Tabu Search metaheuristic for solving Traveling Salesman Problems (TSP), featuring both a modified advanced version and a clean vanilla implementation with detailed performance metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Vanilla Tabu Search](#vanilla-tabu-search)
- [Modified Tabu Search](#modified-tabu-search)
- [Usage Examples](#usage-examples)
- [Performance Metrics](#performance-metrics)
- [TSP Problems](#tsp-problems)
- [Algorithm Details](#algorithm-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides two implementations of Tabu Search for TSP problems:

1. **Vanilla Tabu Search** (`vanilla_tabu.py`) - A clean, original implementation focusing on the core algorithm with comprehensive timing metrics
2. **Modified Tabu Search** (`tsp.py`, `d2d.py`) - An advanced implementation with multiple neighborhoods, parallelization, and multi-objective optimization

### Key Features

- ✅ **Multiple TSP Problem Support** - EUC_2D, ATT distance types
- ✅ **Comprehensive Metrics** - Detailed timing and performance analysis
- ✅ **Visualization** - Solution plotting and metrics visualization
- ✅ **Parallel Processing** - Multi-core support for faster execution
- ✅ **Flexible Parameters** - Configurable iterations, tabu sizes, and stopping criteria
- ✅ **TSPLIB Integration** - Standard TSP problem format support
- ✅ **Command Line Interface** - Easy-to-use CLI for both implementations

## 📁 Project Structure

```
tabu-search/
├── README.md                          # This file
├── vanilla_tabu.py                    # Vanilla tabu search implementation
├── demo_vanilla_tabu.py               # Demonstration script
├── VANILLA_TABU_README.md             # Detailed vanilla tabu documentation
├── tsp.py                             # Modified tabu search for TSP
├── d2d.py                             # Modified tabu search for D2D problems
├── requirements.txt                   # Python dependencies
├── setup.cfg                          # Package configuration
├── pyrightconfig.json                 # Type checking configuration
├── ts/                                # Core tabu search modules
│   ├── __init__.py
│   ├── tsp/                           # TSP-specific implementations
│   ├── d2d/                           # D2D-specific implementations
│   ├── abc/                           # Abstract base classes
│   └── utils/                         # Utility functions
├── problems/                          # TSP problem instances
│   ├── tsp/                           # TSPLIB problems
│   └── d2d/                           # D2D problem instances
├── scripts/                           # Utility scripts
├── tests/                             # Test files
└── extern/                            # External dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sunayana225/capstone-tabu_modified.git
   cd capstone-tabu_modified
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following Python packages:

- `matplotlib` - For plotting solutions and metrics
- `tqdm` - For progress bars
- `numpy` - For numerical computations
- `pybind11` - For C++ extensions (optional)

## 🏃‍♂️ Quick Start

### Vanilla Tabu Search

The simplest way to run the vanilla tabu search:

```bash
# Basic usage with default parameters
python vanilla_tabu.py berlin52

# With custom parameters and verbose output
python vanilla_tabu.py att48 -i 200 -t 15 -v

# Show solution plot
python vanilla_tabu.py eil51 -p

# Run demonstration on multiple problems
python demo_vanilla_tabu.py
```

### Modified Tabu Search

For the advanced implementation:

```bash
# TSP problems
python tsp.py berlin52 -i 500 -t 10 -v

# D2D problems
python d2d.py 6.5.1 -i 1000 -t 15 -v
```

## 🔬 Vanilla Tabu Search

The vanilla implementation provides a clean, educational version of tabu search with detailed metrics.

### Features

- **Original Algorithm**: Classic tabu search with 2-opt neighborhood
- **Comprehensive Metrics**: Execution time, iterations, improvements
- **Visualization**: Solution plots and performance charts
- **Flexible Parameters**: Configurable search parameters

### Command Line Options

```bash
python vanilla_tabu.py <problem> [options]

Options:
  -i, --iterations      Maximum iterations (default: 1000)
  -t, --tabu-size       Tabu list size (default: 10)
  -n, --no-improvement  Max iterations without improvement (default: 100)
  -v, --verbose         Show detailed progress
  -p, --plot            Show solution plot
  -m, --metrics-plot    Show metrics plot
  -d, --dump            Save results to JSON file
```

### Example Output

```
============================================================
VANILLA TABU SEARCH RESULTS
============================================================
Problem: berlin52
Total execution time: 1.0692 seconds
Total iterations: 96
Improvements found: 12
Initial cost: 8980.92
Final best cost: 7938.77
Improvement: 1042.14 (11.60%)
Average time per iteration: 0.011137 seconds
Improvement rate: 12.50%
============================================================
```

## ⚡ Modified Tabu Search

The modified implementation includes advanced features for research and production use.

### Features

- **Multiple Neighborhoods**: Swap, SegmentShift, SegmentReverse
- **Parallel Processing**: Multi-core neighborhood evaluation
- **Multi-objective**: Support for D2D problems
- **Advanced Metrics**: Comprehensive performance tracking
- **Post-optimization**: Additional local search phases

### TSP Usage

```bash
python tsp.py <problem> [options]

Options:
  -i, --iterations      Number of iterations (default: 500)
  -s, --shuffle-after   Shuffle after N non-improved iterations (default: 50)
  -t, --tabu-size       Tabu size for neighborhoods (default: 10)
  -o, --optimal         Read optimal solution from archive
  -v, --verbose         Show progress bar and plot
  -d, --dump            Save solution to file
  --pool-size           Process pool size (default: CPU count)
```

### D2D Usage

```bash
python d2d.py <problem> [options]

Options:
  -i, --iterations           Number of iterations (default: 1500)
  -t, --tabu-size           Tabu size (default: 10)
  -c, --drone-config        Drone configuration index (default: 0)
  -e, --energy-mode         Energy mode: linear, non-linear, endurance
  -k, --propagation-priority Propagation priority strategy
  -m, --max-propagation     Max propagating solutions (default: 5)
  -v, --verbose             Show progress and plots
  -d, --dump                Save results to file
```

## 📊 Performance Metrics

Both implementations provide detailed performance metrics:

### Timing Metrics
- **Total execution time** - Wall-clock time for the entire search
- **Time per iteration** - Average time per search iteration
- **Iterations per second** - Search speed metric

### Search Metrics
- **Total iterations** - Number of search iterations performed
- **Improvements found** - Number of times a better solution was found
- **Tabu moves** - Number of moves made while in tabu status
- **Improvement rate** - Percentage of iterations that found improvements

### Solution Metrics
- **Initial cost** - Cost of the starting solution
- **Final cost** - Cost of the best solution found
- **Improvement** - Absolute and percentage improvement
- **Cost history** - Cost progression throughout the search

## 🗺️ TSP Problems

The project includes support for TSPLIB problems:

### Available Problems

| Problem | Cities | Type | Description |
|---------|--------|------|-------------|
| berlin52 | 52 | EUC_2D | Berlin city tour |
| att48 | 48 | ATT | ATT48 problem |
| eil51 | 51 | EUC_2D | Eil51 problem |
| a280 | 280 | EUC_2D | Large TSP instance |
| pr2392 | 2392 | EUC_2D | Very large instance |

### Problem Formats

- **EUC_2D**: Euclidean 2D coordinates
- **ATT**: Pseudo-Euclidean distance (ATT format)
- **CEIL_2D**: Ceiling of Euclidean distance

## 🔧 Algorithm Details

### Vanilla Tabu Search Algorithm

1. **Initialization**
   - Generate initial solution using nearest neighbor heuristic
   - Initialize empty tabu list
   - Set best solution to initial solution

2. **Main Loop**
   - Generate all 2-opt neighbors
   - Find best non-tabu neighbor
   - Apply aspiration criteria (allow tabu if better than best)
   - Update current solution
   - Add move to tabu list
   - Update best solution if improved

3. **Termination**
   - Stop after maximum iterations
   - Stop if no improvement for specified iterations
   - Return best solution found

### Modified Tabu Search Features

- **Multiple Neighborhoods**: Different move types for exploration
- **Parallel Evaluation**: Multi-core neighborhood evaluation
- **Shuffling**: Random perturbations to escape local optima
- **Post-optimization**: Additional local search phases
- **Multi-objective**: Support for multiple optimization criteria

## 📈 Example Results

### Performance Comparison

| Problem | Algorithm | Initial Cost | Final Cost | Improvement | Time (s) |
|---------|-----------|--------------|------------|-------------|----------|
| berlin52 | Vanilla | 8980.92 | 7938.77 | 11.60% | 1.069 |
| berlin52 | Modified | 8980.92 | 7542.00 | 16.02% | 2.145 |
| att48 | Vanilla | 12861.00 | 11118.00 | 13.55% | 0.344 |
| att48 | Modified | 12861.00 | 10628.00 | 17.36% | 0.892 |

### Key Insights

- **Vanilla Tabu Search**: Faster execution, good for understanding the algorithm
- **Modified Tabu Search**: Better solution quality, suitable for research/production
- **Both implementations**: Provide comprehensive metrics for analysis

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/tsp_test.py
python -m pytest tests/d2d_test.py
```

## 📚 Documentation

- [Vanilla Tabu Search Documentation](VANILLA_TABU_README.md) - Detailed vanilla implementation guide
- [TSPLIB Documentation](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) - TSP problem format
- [Tabu Search Theory](https://en.wikipedia.org/wiki/Tabu_search) - Algorithm background

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run linting
python -m black .
python -m flake8 .

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sunayana** - *Initial work* - [Sunayana225](https://github.com/Sunayana225)
- **Shrinidhi**-[Shrinidhigans](https://github.com/Shrinidhigans)

## 🙏 Acknowledgments

- TSPLIB for providing standard TSP problem instances
- The tabu search research community for algorithm development
- Contributors and users who provided feedback and improvements

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Sunayana225/capstone-tabu_modified/issues) page
2. Create a new issue with detailed information
3. Contact: yakkalasunayana1605@gmail.com

---

**Happy Optimizing! 🚀**
