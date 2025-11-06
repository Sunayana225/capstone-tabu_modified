# 🔍 Advanced Tabu Search Optimization Platform

> **A comprehensive web-based platform for Tabu Search optimization featuring interactive dashboards, algorithm comparison tools, and AI-powered explanations.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 **What is This Platform?**

This is a **modern, web-based optimization platform** that showcases the power of **Tabu Search** metaheuristic algorithms through:

- �️ **Interactive Web Interface** - Beautiful Streamlit dashboards  
- 📊 **Algorithm Comparison** - Compare Tabu Search vs Genetic Algorithm, Ant Colony, Simulated Annealing
- 🤖 **AI Explanations** - Google Gemini integration for plain-English algorithm explanations
- 🔬 **Educational Tool** - Perfect for learning optimization algorithms
- ⚡ **Production Ready** - FastAPI backend with professional architecture

---

## 🚀 **Quick Start - Get Running in 3 Steps**

### **Step 1: Clone & Setup**
```bash
git clone https://github.com/Sunayana225/capstone-tabu_modified.git
cd "capstone-tabu_modified"
pip install -r requirements.txt
```

### **Step 2: Start the System** 
```bash
# Start API Server (Terminal 1)
.\run_api.ps1

# Start Main Interface (Terminal 2) 
.\run_ui.ps1
```

### **Step 3: Open Your Browser**
- **Main Interface**: http://localhost:8501
- **Algorithm Comparison**: http://localhost:8502 (via comparison dashboard)
- **API Documentation**: http://localhost:8000/docs

**🎉 That's it! You're now running a professional optimization platform!**

---

## 📋 **Table of Contents**

- [Platform Features](#-platform-features)
- [System Architecture](#️-system-architecture)  
- [Installation Guide](#-installation-guide)
- [Usage Guide](#-usage-guide)
- [Algorithm Details](#-algorithm-details)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## � **Platform Features**

### **🖥️ Web-Based Interface**
- **Streamlit Dashboard** - Modern, responsive web interface
- **Real-time Visualization** - Interactive charts and solution plots
- **Parameter Tuning** - Easy-to-use controls for algorithm parameters
- **Progress Tracking** - Live updates during optimization

### **📊 Algorithm Comparison Suite**
- **Tabu Search** (Vanilla & Modified versions)
- **Genetic Algorithm** - Population-based evolutionary approach
- **Ant Colony Optimization** - Pheromone-based swarm intelligence  
- **Simulated Annealing** - Temperature-based probabilistic search
- **Side-by-side Comparison** - Performance metrics and visualizations

### **🤖 AI-Powered Explanations**
- **Google Gemini Integration** - Generate plain-English explanations
- **Educational Insights** - Understand what algorithms are doing
- **Beginner-Friendly** - Complex optimization made simple

### **⚡ Professional Architecture**
- **FastAPI Backend** - High-performance REST API
- **Modular Design** - Clean, maintainable codebase
- **TSPLIB Support** - Standard benchmark problems
- **Extensible Framework** - Easy to add new algorithms

## 🏗️ **System Architecture**

```
🌐 Web Platform Architecture
┌─────────────────────────────────────────────────┐
│  Frontend Layer (Streamlit)                    │
├─────────────────────────────────────────────────┤
│  � Main UI (Port 8501)                        │
│  │  ├── Tabu Search Interface                  │
│  │  ├── Parameter Controls                     │
│  │  ├── Visualization Dashboard                │
│  │  └── AI Explanation Panel                   │
│  │                                             │
│  📊 Comparison UI (Port 8502)                  │
│     ├── Multi-Algorithm Dashboard              │
│     ├── Performance Comparison                 │
│     └── Interactive Charts                     │
└─────────────────────────────────────────────────┘
                      ↕️ HTTP/REST API
┌─────────────────────────────────────────────────┐
│  Backend Layer (FastAPI - Port 8000)           │
├─────────────────────────────────────────────────┤
│  🔌 API Endpoints                               │
│  │  ├── /solve-tsp (Tabu Search)               │
│  │  ├── /solve-tsp-vanilla                     │
│  │  ├── /solve-tsp-algorithm (GA, ACO, SA)     │
│  │  ├── /compare-algorithms                    │
│  │  └── /available-problems                    │
│  │                                             │
│  🧠 Algorithm Wrappers                         │
│  │  ├── solver_wrapper.py (Tabu Search)        │
│  │  ├── capstone_wrapper.py (Other Algorithms) │
│  │  └── gemini_helper.py (AI Explanations)     │
└─────────────────────────────────────────────────┘
                      ↕️ Direct Integration  
┌─────────────────────────────────────────────────┐
│  Algorithm Layer                                │
├─────────────────────────────────────────────────┤
│  � Tabu Search Engine                          │
│  │  ├── vanilla_tabu.py (Educational)          │
│  │  ├── tsp.py (Advanced Multi-Neighborhood)   │
│  │  └── d2d.py (Device-to-Device Problems)     │
│  │                                             │
│  🧬 Comparison Algorithms                       │
│  │  ├── genetic-algo.py                        │
│  │  ├── ant-colony-opt.py                      │
│  │  └── simulated-annealing.py                 │
│  │                                             │
│  📚 Problem Database (TSPLIB)                   │
│     ├── berlin52, att48, eil51...               │
│     └── 100+ TSP benchmark instances            │
└─────────────────────────────────────────────────┘
```

### **📁 Directory Structure**
```
📦 capstone-tabu_modified/
├── 🚀 run_api.ps1              # Start API server  
├── 🚀 run_ui.ps1               # Start main interface
├── 🚀 run_comparison.bat       # Start comparison dashboard
├── 📄 requirements.txt         # Python dependencies (everything you need)
├── 📄 README.md               # This comprehensive documentation
├── 📁 app/                    # Web application layer
│   ├── ui.py                  # Main Streamlit interface  
│   ├── comparison_ui.py       # Algorithm comparison dashboard
│   ├── api.py                 # FastAPI backend server
│   ├── solver_wrapper.py      # Tabu search integration
│   ├── capstone_wrapper.py    # Multi-algorithm wrapper
│   └── gemini_helper.py       # AI explanation generator
├── 📁 tabu-search/            # Core tabu search algorithms
│   ├── vanilla_tabu.py        # Educational implementation
│   ├── tsp.py                 # Advanced multi-neighborhood  
│   ├── ts/                    # Algorithm framework
│   └── problems/              # TSPLIB problem instances
├── 📁 Capstone/               # Comparison algorithms
│   ├── genetic-algo.py        # Genetic algorithm
│   ├── ant-colony-opt.py      # Ant colony optimization
│   └── simulated-annealing.py # Simulated annealing
└── 📁 .streamlit/             # Configuration & secrets
    └── secrets.toml           # API keys (Gemini)
```

## �️ **Installation Guide**

### **📋 Prerequisites**
- **Python 3.11+** (recommended) or Python 3.8+
- **pip package manager** 
- **Git** (for cloning repository)
- **Windows PowerShell** (for launch scripts)

### **⚡ Quick Installation**

```bash
# 1. Clone the repository
git clone https://github.com/Sunayana225/capstone-tabu_modified.git
cd capstone-tabu_modified

# 2. Install all dependencies  
pip install -r requirements-app.txt

# 3. Optional: Set up AI explanations (Gemini)
# Get free API key from: https://makersuite.google.com/app/apikey
setx GEMINI_API_KEY "your_api_key_here"

# 4. Launch the platform
.\run_api.ps1     # Start API server (Terminal 1)
.\run_ui.ps1      # Start web interface (Terminal 2)
```

### **🔧 Detailed Setup**

#### **Option 1: Standard Installation**
```bash
# Install all dependencies (includes web app + algorithms + AI)
pip install -r requirements.txt
```

# Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv tabu_env

# Activate environment
# Windows:
tabu_env\Scripts\activate
# macOS/Linux:  
source tabu_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **📦 Core Dependencies**

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28.0 | Web interface framework |
| `fastapi` | =0.115.0 | REST API backend |
| `plotly` | ≥5.17.0 | Interactive visualizations |
| `pandas` | ≥2.0.0 | Data manipulation |
| `matplotlib` | ≥3.7.0 | Static plotting |
| `google-generativeai` | ≥0.8.0 | AI explanations (optional) |

### **🤖 AI Explanation Setup (Optional)**

To enable AI-powered explanations:

1. **Get Gemini API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Set Environment Variable**:
   ```bash
   # Windows
   setx GEMINI_API_KEY "your_api_key_here"
   
   # macOS/Linux
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. **Or create `.streamlit/secrets.toml`**:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

### **✅ Verify Installation**
```bash
# Test API server
python -c "import streamlit, fastapi, plotly; print('✅ All packages installed')"

# Test AI integration (optional)
python -c "import google.generativeai; print('✅ Gemini AI ready')"
```

## � **Usage Guide**

### **🚀 Starting the Platform**

#### **Method 1: PowerShell Scripts (Recommended)**
```bash
# Terminal 1: Start API Server
.\run_api.ps1

# Terminal 2: Start Main Interface  
.\run_ui.ps1

# Terminal 3: Start Comparison Dashboard (Optional)
.\run_comparison.bat
```

#### **Method 2: Manual Commands**
```bash
# Start API Server (Port 8000)
uvicorn app.api:app --reload --port 8000

# Start Main Interface (Port 8501)
streamlit run app/ui.py --server.port 8501

# Start Comparison Dashboard (Port 8502)  
streamlit run app/comparison_ui.py --server.port 8502
```

### **🖥️ Using the Web Interface**

#### **Main Interface (http://localhost:8501)**
1. **Select TSP Problem**: Choose from 100+ TSPLIB instances
2. **Configure Parameters**: 
   - Iterations: 100-2000 (default: 500)
   - Tabu Size: 5-50 (default: 10)  
   - Algorithm Type: Vanilla vs Modified
3. **Enable AI Explanations**: Check "Generate plain-English explanation"
4. **Run Optimization**: Click "Solve TSP Problem"
5. **View Results**: Interactive charts, solution plots, performance metrics

#### **Comparison Dashboard (http://localhost:8502)**
1. **Navigate to "Algorithm Comparison"**
2. **Select Algorithms**: Choose from GA, ACO, SA, Tabu Search
3. **Configure Each Algorithm**: Set population size, iterations, etc.
4. **Run Comparison**: Click "Run TSP Comparison"  
5. **Analyze Results**: Side-by-side performance charts

### **📊 Understanding Results**

#### **Performance Metrics**
- **Cost**: Total distance (lower = better)
- **Time**: Execution time in milliseconds
- **Iterations**: Number of search iterations performed
- **Improvements**: How many times better solutions were found

#### **Visualizations**
- **Solution Plot**: Shows the optimal tour path
- **Cost History**: Progress during optimization
- **Comparison Charts**: Algorithm performance side-by-side

### **🔍 Advanced Usage**

#### **Command Line Interface** (For developers)
```bash
# Navigate to algorithm directory
cd tabu-search

# Run vanilla tabu search
python vanilla_tabu.py berlin52 -i 1000 -t 15 -v -p

# Run advanced tabu search  
python tsp.py att48 -i 500 -t 10 --pool-size 4 -v

# Parameters:
# -i, --iterations     Number of iterations (default: 500)
# -t, --tabu-size      Tabu list size (default: 10)
# -v, --verbose        Show progress and plots
# -p, --plot           Display solution visualization
# -d, --dump           Save results to JSON file
```

#### **API Endpoints** (For integration)
```python
import requests

# Solve TSP problem via API
response = requests.post("http://localhost:8000/solve-tsp", json={
    "problem": "berlin52",
    "iterations": 500,
    "tabu_size": 10,
    "verbose": False
})

result = response.json()
print(f"Best cost: {result['solution']['cost']}")
```

## 🧠 **Algorithm Details**

### **🔍 Tabu Search Overview**

Tabu Search is a metaheuristic algorithm that guides local search procedures to explore solution spaces beyond local optimality. It uses **memory structures** (tabu lists) to avoid cycling and encourage exploration of new regions.

#### **Core Concepts**
- **Memory-Based Search**: Remembers recent moves to avoid cycling
- **Aspiration Criteria**: Overrides tabu restrictions for exceptional solutions  
- **Intensification & Diversification**: Balances local improvement with exploration
- **Neighborhood Exploration**: Systematically examines solution modifications

### **🎯 Algorithm Implementations**

#### **1. Vanilla Tabu Search** (`vanilla_tabu.py`)
*Educational implementation focusing on core algorithm clarity*

```python
# Key Features
✅ 2-opt Neighborhood Structure
✅ Fixed-Size Tabu List (FIFO)
✅ Simple Aspiration Criteria
✅ Comprehensive Performance Metrics
✅ Visualization & Plotting Support

# Best For: Learning, Quick Testing, Understanding Fundamentals
```

**Algorithm Flow:**
```
1. Generate initial solution (Nearest Neighbor)
2. For each iteration:
   ├── Generate all 2-opt neighbors
   ├── Find best non-tabu neighbor  
   ├── Apply aspiration if move improves best solution
   ├── Update current solution & tabu list
   └── Track performance metrics
3. Return best solution found
```

#### **2. Advanced Tabu Search** (`tsp.py`)
*Production-ready implementation with advanced features*

```python
# Advanced Features  
✅ Multiple Neighborhood Types (Swap, SegmentShift, SegmentReverse)
✅ Parallel Neighborhood Evaluation
✅ Adaptive Shuffling Mechanism
✅ Post-Optimization Local Search
✅ Multi-Core Processing Support

# Best For: Research, Production Systems, Complex Problems
```

**Enhanced Algorithm Flow:**
```
1. Initialize with nearest neighbor heuristic
2. Main tabu search loop:
   ├── Evaluate multiple neighborhoods in parallel
   ├── Apply tabu restrictions per neighborhood
   ├── Select best non-tabu move across all neighborhoods
   ├── Update solution and multiple tabu lists
   └── Apply shuffling if stagnation detected
3. Post-optimization refinement
4. Return optimized solution
```

### **🏆 Comparison Algorithms**

#### **🧬 Genetic Algorithm** (`genetic-algo.py`)
- **Population-Based**: Maintains multiple solutions simultaneously
- **Crossover & Mutation**: Combines good solutions, introduces variation
- **Selection Pressure**: Favors better solutions for reproduction
- **Best For**: Complex landscapes, global optimization

#### **🐜 Ant Colony Optimization** (`ant-colony-opt.py`)  
- **Swarm Intelligence**: Multiple agents (ants) construct solutions
- **Pheromone Trails**: Indirect communication guides search
- **Probabilistic Construction**: Solutions built step-by-step
- **Best For**: Path-finding problems, dynamic environments

#### **🌡️ Simulated Annealing** (`simulated-annealing.py`)
- **Temperature-Based**: Accepts worse moves with decreasing probability
- **Cooling Schedule**: Gradually reduces acceptance of bad moves
- **Single Solution**: Maintains one current solution
- **Best For**: Continuous optimization, avoiding local optima

### **📊 Algorithm Comparison Matrix**

| Feature | Tabu Search | Genetic Algorithm | Ant Colony | Simulated Annealing |
|---------|-------------|-------------------|------------|-------------------|
| **Memory Usage** | ✅ Explicit (Tabu List) | ✅ Implicit (Population) | ✅ Pheromone Matrix | ❌ No Memory |
| **Population** | Single Solution | Multiple Solutions | Multiple Agents | Single Solution |
| **Deterministic** | ✅ Mostly | ❌ Stochastic | ❌ Probabilistic | ❌ Probabilistic |
| **Parameter Sensitivity** | 🟡 Medium | 🔴 High | 🔴 High | 🟡 Medium |
| **Convergence Speed** | ✅ Fast | 🟡 Medium | 🟡 Medium | 🔴 Slow |
| **Solution Quality** | ✅ High | ✅ High | 🟡 Good | 🟡 Good |
| **Scalability** | ✅ Excellent | 🟡 Good | 🟡 Good | ✅ Excellent |

### **🎯 TSP Problem Support**

#### **TSPLIB Integration**
- **100+ Benchmark Problems**: From 14 to 85,900 cities
- **Multiple Distance Types**: EUC_2D, ATT, CEIL_2D, GEO
- **Optimal Solutions**: Known best solutions for validation
- **Standard Format**: Industry-recognized problem instances

#### **Popular Test Problems**
| Problem | Cities | Optimal Cost | Difficulty | Description |
|---------|--------|--------------|------------|-------------|
| `berlin52` | 52 | 7,542 | 🟢 Easy | Berlin city locations |
| `att48` | 48 | 10,628 | 🟢 Easy | ATT distance metric |
| `eil51` | 51 | 426 | 🟢 Easy | Christofides & Eilon |
| `a280` | 280 | 2,579 | 🟡 Medium | Large instance |
| `pr2392` | 2,392 | 378,032 | 🔴 Hard | Very large instance |

### **⚡ Performance Optimization**

#### **Parallel Processing**
- **Multi-Core Support**: Utilizes all available CPU cores
- **Neighborhood Parallelization**: Evaluates neighborhoods simultaneously  
- **Process Pool**: Efficient task distribution
- **Automatic Scaling**: Adapts to system capabilities

#### **Memory Management**  
- **Efficient Data Structures**: Optimized for speed and memory
- **Tabu List Optimization**: Fixed-size circular buffers
- **Solution Caching**: Avoids redundant calculations
- **Garbage Collection**: Proper memory cleanup

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

## 🔌 **API Reference**

### **FastAPI Backend Endpoints**

The platform provides a RESTful API for programmatic access:

#### **Core Endpoints**

```http
GET  /health                    # API health check
GET  /available-problems        # List all TSP problems  
POST /solve-tsp                 # Run advanced tabu search
POST /solve-tsp-vanilla         # Run vanilla tabu search
POST /solve-tsp-algorithm       # Run comparison algorithms
POST /compare-tsp-algorithms    # Multi-algorithm comparison
```

#### **Example API Usage**

```python
import requests

# Solve TSP with advanced tabu search
response = requests.post("http://localhost:8000/solve-tsp", json={
    "problem": "berlin52",
    "iterations": 500,
    "shuffle_after": 50,
    "tabu_size": 10,
    "pool_size": 4,
    "verbose": False
})

result = response.json()
print(f"Solution cost: {result['solution']['cost']}")
print(f"Execution time: {result['elapsed_ms']}ms")

# Compare multiple algorithms
comparison = requests.post("http://localhost:8000/compare-tsp-algorithms", json={
    "algorithms": ["genetic_algorithm", "ant_colony_optimization", "tabu_search"],
    "problem": "att48",
    "runs": 3
})

results = comparison.json()
for algo, data in results['algorithms'].items():
    print(f"{algo}: {data['solution']['cost']}")
```

#### **Request/Response Schemas**

```typescript
// TSP Solve Request
interface TSPRequest {
    problem: string;           // TSP problem name
    iterations?: number;       // Default: 500
    tabu_size?: number;       // Default: 10  
    shuffle_after?: number;   // Default: 50
    pool_size?: number;       // Default: CPU count
    verbose?: boolean;        // Default: false
}

// TSP Response
interface TSPResponse {
    problem: string;
    solution: {
        cost: number;
        path: number[];
    };
    parameters: TSPRequest;
    elapsed_ms: number;
    plot_base64?: string;     // If verbose=true
}
```

### **🌐 Web Interface Endpoints**

- **Main Interface**: http://localhost:8501
  - Interactive TSP solver
  - Parameter configuration
  - AI explanations  
  - Real-time visualization

- **Comparison Dashboard**: http://localhost:8502  
  - Multi-algorithm comparison
  - Performance analysis
  - Side-by-side charts

- **API Documentation**: http://localhost:8000/docs
  - Interactive Swagger UI
  - Endpoint testing
  - Schema documentation

## � **Troubleshooting**

### **Common Issues & Solutions**

#### **❌ Port Already in Use**
```bash
# Error: Address already in use: port 8501
# Solution: Kill existing processes
taskkill /f /im python.exe
# Or use different ports:
streamlit run app/ui.py --server.port 8503
```

#### **❌ Module Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'streamlit'
# Solution: Install dependencies
pip install -r requirements.txt

# Error: No module named 'ts'
# Solution: Install tabu-search dependencies  
cd tabu-search && pip install -r requirements.txt
```

#### **❌ API Connection Failed**
```bash
# Error: Connection refused to localhost:8000
# Solution: Start API server first
.\run_api.ps1
# Wait for "Uvicorn running on http://127.0.0.1:8000"
```

#### **❌ Gemini API Errors**
```bash
# Error: Gemini SDK not installed
pip install google-generativeai

# Error: Invalid API key
# Solution: Set correct environment variable
setx GEMINI_API_KEY "your_actual_api_key"
```

### **Performance Tips**

#### **🚀 Speed Optimization**
- **Reduce Iterations**: Start with 100-200 for testing
- **Smaller Problems**: Use berlin52, att48 for quick tests
- **Disable Verbose**: Turn off progress bars for faster execution
- **Use Pool Size**: Set `--pool-size` to your CPU core count

#### **💾 Memory Management**
- **Large Problems**: Increase system virtual memory for 1000+ city problems
- **Multiple Runs**: Close browser tabs between comparison runs
- **Clear Cache**: Restart servers if memory usage grows

### **🔍 Debugging**

#### **Enable Debug Mode**
```bash
# API Server Debug
uvicorn app.api:app --reload --log-level debug

# Streamlit Debug  
streamlit run app/ui.py --logger.level debug
```

#### **Check Logs**
```bash
# View API logs
tail -f uvicorn.log

# Check Python errors
python -u app/ui.py 2>&1 | tee streamlit.log
```

### **📞 Getting Help**

- **GitHub Issues**: [Report bugs & request features](https://github.com/Sunayana225/capstone-tabu_modified/issues)
- **Documentation**: Check this README and `COMPARISON_README.md`
- **API Docs**: Visit http://localhost:8000/docs when server is running

## 🤝 **Contributing**

We welcome contributions! Here's how to get involved:

### **🛠️ Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/your-username/capstone-tabu_modified.git
cd capstone-tabu_modified

# Create development branch
git checkout -b feature/your-feature-name

# Install all dependencies
pip install -r requirements.txt
pip install -r tabu-search/requirements.txt

# Make your changes and test
python -m pytest tabu-search/tests/
```

### **📝 Contribution Areas**
- 🔍 **New Algorithms**: Add other metaheuristics (PSO, DE, etc.)
- 🎨 **UI Improvements**: Enhance Streamlit interface
- 📊 **Visualizations**: Create new chart types
- 🧪 **Test Coverage**: Add unit tests
- 📚 **Documentation**: Improve guides and examples
- 🐛 **Bug Fixes**: Fix issues and optimize performance

### **📋 Pull Request Process**
1. **Create Issue**: Describe the feature/fix
2. **Fork & Branch**: Work on a feature branch
3. **Test**: Ensure all tests pass
4. **Document**: Update README if needed
5. **Submit PR**: Provide clear description

### **🎯 Code Style**
- **Python**: Follow PEP 8 guidelines
- **Type Hints**: Use type annotations
- **Docstrings**: Document functions and classes  
- **Comments**: Explain complex logic

## � **License & Credits**

### **📜 License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Free for commercial and personal use
✅ Commercial use    ✅ Modification    ✅ Distribution    ✅ Private use
```

### **👥 Authors**
- **[Sunayana](https://github.com/Sunayana225)** - *Principal Developer & Researcher*
  - Core algorithm implementation
  - Web platform architecture  
  - Performance optimization

### **🙏 Acknowledgments**
- **TSPLIB** - Providing standard benchmark problems
- **Streamlit Team** - Amazing web framework for Python
- **FastAPI** - High-performance API framework
- **Google Gemini** - AI-powered explanations
- **Research Community** - Tabu search algorithm development
- **Open Source Contributors** - Libraries and tools used

### **📚 References**
- Glover, F. (1986). "Future paths for integer programming and links to artificial intelligence"
- Gendreau, M., & Potvin, J. Y. (2010). "Handbook of metaheuristics"
- TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- Tabu Search: https://en.wikipedia.org/wiki/Tabu_search

---

## 🎉 **What's Next?**

### **🚀 Quick Actions**
1. **⭐ Star this repository** if you found it useful
2. **🍴 Fork it** to create your own version  
3. **📝 Try the examples** in the Usage Guide
4. **🐛 Report issues** if you find any bugs
5. **💡 Suggest features** for future improvements

### **📈 Roadmap**
- 🔮 **Multi-Objective Optimization** - NSGA-II, SPEA2 integration
- 🌐 **Cloud Deployment** - Docker containerization, cloud hosting
- 📱 **Mobile Interface** - Responsive design for tablets/phones
- 🤖 **Advanced AI** - GPT integration, automated parameter tuning
- 📊 **Analytics Dashboard** - Usage statistics, performance tracking

### **🎓 Educational Use**
Perfect for:
- **Computer Science Courses** - Algorithm analysis and implementation
- **Operations Research** - Optimization methods and metaheuristics  
- **Research Projects** - Benchmark testing and algorithm development
- **Industry Training** - Learning optimization techniques

---

<div align="center">

### **🌟 Built with ❤️ for the Optimization Community**

**[🏠 Homepage](https://github.com/Sunayana225/capstone-tabu_modified) • [📖 Documentation](README.md) • [🐛 Issues](https://github.com/Sunayana225/capstone-tabu_modified/issues) • [💬 Discussions](https://github.com/Sunayana225/capstone-tabu_modified/discussions)**

*Happy Optimizing! 🚀*

</div>




