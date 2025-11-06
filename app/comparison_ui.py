"""
Algorithm Comparison Dashboard for Tabu Search vs Other Optimization Algorithms
Shows comprehensive comparison of all Capstone algorithms
"""

from __future__ import annotations

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional

API_BASE = "http://127.0.0.1:8000"


def _load_tsp_problems() -> List[str]:
    """Load all available TSP problems from the problems directory."""
    try:
        from pathlib import Path
        # Get the root directory (parent of app folder)
        root = Path(__file__).resolve().parents[1]
        tsp_dir = root / "tabu-search" / "problems" / "tsp"
        names: List[str] = []
        if tsp_dir.exists():
            for child in tsp_dir.iterdir():
                if child.is_dir() and child.name.endswith('.tsp'):
                    # Look for a file with the same name as the directory
                    tsp_file = child / child.name
                    if tsp_file.exists():
                        # The problem name is the directory name without .tsp extension
                        problem_name = child.name.replace('.tsp', '')
                        names.append(problem_name)
        return sorted(names)  # Return all problems without limit
    except Exception as e:
        print(f"Error loading TSP problems: {e}")
        return ["eil51", "berlin52", "a280", "att48", "gr17", "fri26", "bayg29"]


def display_base64_image(base64_str: str, caption: str = "", width: int = 600):
    """Display base64 encoded image in Streamlit."""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        st.image(image, caption=caption, width=width)
    except Exception as e:
        st.error(f"Error displaying image: {e}")


def create_comparison_chart(results: Dict[str, Any]) -> Optional[go.Figure]:
    """Create an interactive comparison chart of algorithm performance."""
    
    # Extract data for plotting
    algorithms = []
    costs = []
    times = []
    colors_cost = []
    colors_time = []
    
    for alg_name, result in results.items():
        if "error" not in result and "solution" in result:
            algorithm_display = result.get("algorithm", alg_name)
            algorithms.append(algorithm_display)
            costs.append(result["solution"]["cost"])
            times.append(result["elapsed_ms"])
            
            # Use different colors for Tabu Search to make it stand out
            if "tabu" in alg_name.lower() or "tabu" in algorithm_display.lower():
                colors_cost.append("#FF6B6B")  # Red for Tabu Search
                colors_time.append("#FF6B6B")
            else:
                colors_cost.append("lightblue")
                colors_time.append("lightcoral")
    
    if not algorithms:
        return None
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Solution Cost (Lower is Better)', 'Execution Time (ms)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cost comparison
    fig.add_trace(
        go.Bar(
            x=algorithms,
            y=costs,
            name="Cost",
            marker_color=colors_cost,
            text=[f"{c:.2f}" for c in costs],
            textposition="auto",
            textfont=dict(size=12, color="white")
        ),
        row=1, col=1
    )
    
    # Time comparison
    fig.add_trace(
        go.Bar(
            x=algorithms,
            y=times,
            name="Time (ms)",
            marker_color=colors_time,
            text=[f"{t:.1f}" for t in times],
            textposition="auto",
            textfont=dict(size=12, color="white")
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Algorithm Performance Comparison",
        showlegend=False,
        height=500,
        font=dict(size=12)
    )
    
    fig.update_xaxes(title_text="Algorithm", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Algorithm", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    
    return fig


def display_algorithm_results(results: Dict[str, Any], title: str):
    """Display results for a set of algorithms."""
    
    st.subheader(title)
    
    if not results:
        st.warning("No results to display.")
        return
    
    # Create metrics columns
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        st.error("All algorithms failed to run.")
        for alg, result in results.items():
            if "error" in result:
                st.error(f"**{alg}**: {result['error']}")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    costs = [r["solution"]["cost"] for r in valid_results.values()]
    times = [r["elapsed_ms"] for r in valid_results.values()]
    
    with col1:
        st.metric("Best Cost", f"{min(costs):.2f}")
    with col2:
        st.metric("Worst Cost", f"{max(costs):.2f}")
    with col3:
        st.metric("Avg Time", f"{sum(times)/len(times):.1f} ms")
    
    # Create comparison chart
    fig = create_comparison_chart(valid_results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results for each algorithm
    for alg_name, result in valid_results.items():
        with st.expander(f"📊 {result.get('algorithm', alg_name)} - Details"):
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Parameters:**")
                if "parameters" in result:
                    for key, value in result["parameters"].items():
                        st.write(f"- {key}: {value}")
                
                st.write("**Results:**")
                if "solution" in result:
                    st.write(f"- Cost: {result['solution']['cost']:.2f}")
                    st.write(f"- Time: {result['elapsed_ms']:.1f} ms")
                
                if "comparison" in result:
                    st.write("**Comparison:**")
                    comp = result["comparison"]
                    st.write(f"- Gap to best: {comp['gap_to_best']:.1f}%")
                    st.write(f"- Cost ratio: {comp['cost_ratio']:.2f}x")
            
            with col2:
                if "plot_base64" in result:
                    display_base64_image(
                        result["plot_base64"], 
                        f"{result.get('algorithm', alg_name)} Solution",
                        width=400
                    )


def run_tsp_comparison_page():
    """TSP Algorithms Comparison Page."""
    
    st.title("🏆 TSP Algorithm Comparison")
    st.markdown("Compare Tabu Search against other metaheuristic algorithms on TSP problems.")
    
    with st.sidebar:
        st.markdown("### TSP Comparison Settings")
        
        # Problem selection
        st.markdown("**Problem Selection:**")
        
        # Get available problems - use local directory scan for TSP problems
        tsp_problems = _load_tsp_problems()
        
        # Get D2D problems via API if needed
        d2d_problems = []
        try:
            response = requests.get(f"{API_BASE}/available-problems", timeout=10)
            if response.status_code == 200:
                problem_data = response.json()
                d2d_problems = problem_data.get("d2d_problems", [])
        except:
            d2d_problems = []
        
        # Problem type selection
        problem_type = st.radio(
            "Problem Type:",
            ["TSP (Traveling Salesman)", "D2D (Door-to-Door)"],
            help="Choose between TSP or Door-to-Door delivery problems"
        )
        
        if problem_type == "TSP (Traveling Salesman)":
            available_problems = tsp_problems
            selected_problem = st.selectbox(
                "Choose TSP problem:",
                available_problems,
                index=0,
                help="Select a TSP problem instance to solve"
            )
            problem_category = "tsp"
        else:
            if not d2d_problems:
                st.warning("No D2D problems available. Showing TSP problems instead.")
                available_problems = tsp_problems
                selected_problem = st.selectbox(
                    "Choose problem:",
                    available_problems,
                    index=0
                )
                problem_category = "tsp"
            else:
                available_problems = d2d_problems
                selected_problem = st.selectbox(
                    "Choose D2D problem:",
                    available_problems,
                    index=0,
                    help="Select a D2D problem instance to solve"
                )
                problem_category = "d2d"
        
        st.caption(f"Selected: {selected_problem} ({problem_category.upper()})")
        
        # Show problem info
        if problem_category == "d2d":
            st.info(f"📦 D2D Problem: {selected_problem}")
            st.markdown("- Multi-objective optimization (cost + time)")
            st.markdown("- Drone and technician coordination")
            st.markdown("- Only Tabu Search comparison available")
        else:
            st.info(f"🗺️ TSP Problem: {selected_problem}")
            st.markdown("- Single objective (minimize tour distance)")
            st.markdown("- All algorithms available for comparison")
        
        # Algorithm selection (conditional on problem type)
        if problem_category == "tsp":
            algorithms = st.multiselect(
                "Select algorithms to compare:",
                ["genetic_algorithm", "ant_colony_optimization", "simulated_annealing"],
                default=["genetic_algorithm", "ant_colony_optimization", "simulated_annealing"],
                help="Choose TSP algorithms to compare"
            )
            
            # Add Tabu Search option - default to True
            st.markdown("### 🔍 Tabu Search Options")
            include_tabu = st.checkbox("✅ Include Tabu Search algorithms", value=True, 
                                     help="Add both Modified and Vanilla Tabu Search to comparison")
            
            if include_tabu:
                st.success("🎯 **Tabu Search will be highlighted in RED** in comparison charts!")
                st.markdown("**Tabu Search Variants:**")
                st.markdown("- 🔴 **Tabu Search (Modified)** - Enhanced with multiple neighborhoods")
                st.markdown("- 🔴 **Tabu Search (Vanilla)** - Standard implementation")
                tabu_algorithm = st.selectbox("Tabu variant:", ["both", "modified", "vanilla"], 
                                            help="Choose tabu search implementation")
            else:
                st.warning("⚠️ Tabu Search disabled - only Capstone algorithms will be compared")
                tabu_algorithm = None
        else:
            # For D2D problems, only tabu search is available
            st.markdown("**Available Algorithms:**")
            algorithms = []
            include_tabu = True
            tabu_algorithm = "modified"
            st.info("🔍 D2D problems currently support Tabu Search only")
            st.markdown("- Modified Tabu Search with D2D-specific neighborhoods")
            st.markdown("- Multi-objective optimization")
            st.markdown("- Specialized for drone-technician coordination")
        
        st.markdown("### Algorithm Parameters")
        
        # GA Parameters
        if "genetic_algorithm" in algorithms:
            st.markdown("**Genetic Algorithm:**")
            ga_pop = st.slider("Population size", 50, 200, 100)
            ga_gen = st.slider("Generations", 100, 500, 200)
        
        # ACO Parameters  
        if "ant_colony_optimization" in algorithms:
            st.markdown("**Ant Colony Optimization:**")
            aco_ants = st.slider("Number of ants", 10, 50, 30)
            aco_iter = st.slider("ACO Iterations", 50, 300, 150)
        
        # SA Parameters
        if "simulated_annealing" in algorithms:
            st.markdown("**Simulated Annealing:**")
            sa_temp = st.slider("Initial temperature", 50, 200, 100)
            sa_iter = st.slider("SA Iterations", 1000, 10000, 5000)
    
    # Dynamic button text based on problem type
    button_text = f"🚀 Run {problem_category.upper()} Comparison"
    
    if st.button(button_text, type="primary"):
        
        if not algorithms and not include_tabu:
            st.error("Please select at least one algorithm to compare.")
            return
        
        results = {}
        
        comparison_title = f"{problem_category.upper()} Algorithm Comparison on {selected_problem}"
        
        with st.spinner(f"Running {problem_category.upper()} algorithm comparison..."):
            
            # Run algorithms based on problem type
            if problem_category == "tsp" and (algorithms or include_tabu):
                # Run Capstone algorithms + Tabu Search for TSP
                try:
                    payload = {
                        "algorithms": algorithms,
                        "problem": selected_problem,
                        "verbose": False,
                        "include_tabu": include_tabu  # Always include tabu search when checked
                    }
                    
                    # Debug info
                    if include_tabu:
                        st.info(f"🔍 Running comparison with {len(algorithms)} Capstone algorithms + Tabu Search variants")
                    else:
                        st.info(f"🔍 Running comparison with {len(algorithms)} Capstone algorithms only")
                    
                    # Add progress indicator for large problems
                    if selected_problem in ['d1655', 'd15112', 'd18512', 'd2103', 'fl1400', 'fl1577']:
                        st.warning(f"⏳ Large problem detected: {selected_problem}. This may take 5-20 minutes. Please be patient...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("🚀 Starting algorithm comparison...")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("🔄 Running algorithms...")
                    
                    progress_bar.progress(10)  # Show some initial progress
                    response = requests.post(f"{API_BASE}/compare-tsp-algorithms", json=payload, timeout=1200)  # 20 minutes for large problems
                    progress_bar.progress(100)  # Complete when done
                    status_text.text("✅ Algorithm comparison completed!")
                    
                    if response.status_code == 200:
                        capstone_results = response.json()["data"]
                        results.update(capstone_results["algorithms"])
                    else:
                        st.error(f"Failed to run algorithms: {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timed out! The algorithm is taking longer than expected.")
                    st.info("💡 **Solutions:**")
                    st.info("• Try a smaller TSP problem (e.g., berlin52, att48, eil51)")
                    st.info("• Restart the API server if it's unresponsive")
                    st.info("• Large problems (1000+ cities) may need 10-20 minutes")
                    return
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
                    return
            
            # Run D2D specific algorithms if needed
            if problem_category == "d2d" and include_tabu:
                try:
                    # Use D2D-specific endpoint for tabu search
                    tabu_payload = {
                        "problem": selected_problem,
                        "iterations": 1000,
                        "tabu_size": 10,
                        "shuffle_after": 100,
                        "verbose": False
                    }
                    tabu_response = requests.post(f"{API_BASE}/solve-d2d", json=tabu_payload, timeout=120)
                    
                    if tabu_response.status_code == 200:
                        tabu_data = tabu_response.json()["data"]
                        # Convert to comparison format
                        tabu_result = {
                            "algorithm": tabu_data.get("algorithm", "Tabu Search (D2D)"),
                            "problem": tabu_data["problem"] if "problem" in tabu_data else f"{selected_problem} (D2D)",
                            "solution": tabu_data["solution"],
                            "elapsed_ms": tabu_data["elapsed_ms"],
                            "parameters": tabu_data.get("parameters", {}),
                            "problem_type": problem_category
                        }
                        
                        # Add problem-specific details
                        if "problem_details" in tabu_data:
                            tabu_result["problem_details"] = tabu_data["problem_details"]
                        
                        results["tabu_search_d2d"] = tabu_result
                    else:
                        st.error(f"Failed to run D2D Tabu Search: {tabu_response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"D2D Tabu Search connection error: {e}")
        
        if results:
            display_algorithm_results(results, comparison_title)
            
            # Summary table
            st.subheader("📈 Performance Summary Table")
            
            table_data = []
            for alg_name, result in results.items():
                if "error" not in result and "solution" in result:
                    table_data.append({
                        "Algorithm": result.get("algorithm", alg_name),
                        "Cost": f"{result['solution']['cost']:.2f}",
                        "Time (ms)": f"{result['elapsed_ms']:.1f}",
                        "Gap to Best (%)": f"{result.get('comparison', {}).get('gap_to_best', 0):.1f}"
                    })
            
            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)


def run_job_scheduling_page():
    """Job Scheduling Algorithms Page."""
    
    st.title("📋 Job Scheduling with Tabu Search")
    st.markdown("Compare different Job Scheduling problem formulations using Tabu Search.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Job Scheduling")
        st.markdown("Jobs with deadlines and profits (unit duration)")
        
        if st.button("Run Basic Job Scheduling"):
            with st.spinner("Running basic job scheduling..."):
                try:
                    payload = {
                        "job_type": "basic",
                        "iterations": 50,
                        "tabu_tenure": 5,
                        "neighbors": 20,
                        "verbose": False
                    }
                    response = requests.post(f"{API_BASE}/solve-job-scheduling", json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()["data"]
                        
                        # Display results
                        st.success(f"**Best Profit:** {result['solution']['profit']}")
                        st.write(f"**Best Sequence:** {result['solution']['sequence']}")
                        st.write(f"**Execution Time:** {result['elapsed_ms']:.1f} ms")
                        
                        # Display job details
                        st.write("**Job Details:**")
                        for job in result["jobs"]:
                            st.write(f"Job {job['id']}: Deadline={job['deadline']}, Profit={job['profit']}")
                        
                        # Display visualization
                        if "plot_base64" in result:
                            display_base64_image(result["plot_base64"], "Job Schedule", width=600)
                    else:
                        st.error(f"Error: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
    
    with col2:
        st.subheader("Duration-based Job Scheduling")
        st.markdown("Jobs with variable durations, deadlines, and profits")
        
        if st.button("Run Duration-based Job Scheduling"):
            with st.spinner("Running duration-based job scheduling..."):
                try:
                    payload = {
                        "job_type": "duration",
                        "iterations": 50,
                        "tabu_tenure": 5,
                        "neighbors": 20,
                        "verbose": False
                    }
                    response = requests.post(f"{API_BASE}/solve-job-scheduling", json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()["data"]
                        
                        # Display results
                        st.success(f"**Best Profit:** {result['solution']['profit']}")
                        st.write(f"**Best Sequence:** {result['solution']['sequence']}")
                        st.write(f"**Execution Time:** {result['elapsed_ms']:.1f} ms")
                        
                        # Display job details
                        st.write("**Job Details:**")
                        for job in result["jobs"]:
                            st.write(f"Job {job['id']}: Duration={job['duration']}, Deadline={job['deadline']}, Profit={job['profit']}")
                        
                        # Display schedule
                        st.write("**Schedule:**")
                        for job_id, start, end in result['solution']['schedule']:
                            st.write(f"Job {job_id}: Time {start}-{end}")
                        
                        # Display visualization
                        if "plot_base64" in result:
                            display_base64_image(result["plot_base64"], "Job Schedule with Durations", width=600)
                    else:
                        st.error(f"Error: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")


def run_comprehensive_comparison():
    """Run comprehensive comparison of all algorithms."""
    
    st.title("🔬 Comprehensive Algorithm Analysis")
    st.markdown("Complete comparison of all optimization algorithms in the Capstone collection.")
    
    if st.button("🎯 Run Complete Analysis", type="primary"):
        
        with st.spinner("Running comprehensive algorithm analysis... This may take a few minutes."):
            try:
                response = requests.post(f"{API_BASE}/compare-all-algorithms", params={"verbose": False}, timeout=600)
                
                if response.status_code == 200:
                    data = response.json()["data"]
                    
                    # Display TSP comparison
                    if "tsp_comparison" in data:
                        tsp_data = data["tsp_comparison"]
                        display_algorithm_results(
                            tsp_data["algorithms"], 
                            "🗺️ TSP Algorithm Comparison"
                        )
                    
                    # Display Job Scheduling comparison  
                    if "job_scheduling_comparison" in data:
                        job_data = data["job_scheduling_comparison"]
                        
                        st.subheader("📋 Job Scheduling Algorithm Results")
                        
                        for alg_name, result in job_data["algorithms"].items():
                            with st.expander(f"📊 {result.get('algorithm', alg_name)}"):
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.write(f"**Profit:** {result['solution']['profit']}")
                                    st.write(f"**Time:** {result['elapsed_ms']:.1f} ms")
                                    st.write(f"**Problem:** {result['problem']}")
                                
                                with col2:
                                    if "plot_base64" in result:
                                        display_base64_image(
                                            result["plot_base64"], 
                                            "Schedule Visualization",
                                            width=400
                                        )
                    
                    # Overall summary
                    st.subheader("📊 Analysis Summary")
                    summary = data.get("summary", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Algorithms", summary.get("total_algorithms_tested", 0))
                    with col2:
                        st.metric("TSP Algorithms", summary.get("tsp_algorithms", 0))
                    with col3:
                        st.metric("Job Scheduling", summary.get("job_scheduling_algorithms", 0))
                    
                else:
                    st.error(f"Analysis failed: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")


def main(standalone: bool = True):
    """Main comparison dashboard."""
    
    # Only set page config if running standalone
    if standalone:
        st.set_page_config(
            page_title="Algorithm Comparison Dashboard", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    # Navigation
    st.sidebar.title("🔍 Algorithm Comparison")
    page = st.sidebar.radio(
        "Choose Analysis Type:",
        ["🏆 TSP Comparison", "📋 Job Scheduling", "🔬 Complete Analysis"]
    )
    
    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### About This Dashboard")
        st.markdown(
            "Compare **🔴 Tabu Search** performance against other optimization algorithms:\n\n"
            "**TSP Algorithms:**\n"
            "- Genetic Algorithm\n"
            "- Ant Colony Optimization\n" 
            "- Simulated Annealing\n"
            "- 🔴 **Tabu Search (Modified)** ⭐\n"
            "- 🔴 **Tabu Search (Vanilla)** ⭐\n\n"
            "**Job Scheduling:**\n"
            "- Basic Tabu Search\n"
            "- Duration-based Tabu Search\n\n"
            "🔴 **Tabu Search algorithms are highlighted in red** in all comparison charts!"
        )
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown(
            "1. Select an analysis type\n"
            "2. Configure parameters\n" 
            "3. Run comparison\n"
            "4. Review results and visualizations"
        )
    
    # Route to appropriate page
    if page == "🏆 TSP Comparison":
        run_tsp_comparison_page()
    elif page == "📋 Job Scheduling":
        run_job_scheduling_page()
    else:  # Complete Analysis
        run_comprehensive_comparison()


if __name__ == "__main__":
    main()