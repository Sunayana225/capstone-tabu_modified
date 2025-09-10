#!/usr/bin/env python3
"""
Demonstration script for Vanilla Tabu Search

This script demonstrates the vanilla tabu search implementation on various
TSP problems with different parameters and shows the timing metrics.
"""

import time
from vanilla_tabu import VanillaTabuSearch, load_tsp_problem, print_metrics


def run_demo():
    """Run demonstration of vanilla tabu search on multiple problems"""
    
    print("="*80)
    print("VANILLA TABU SEARCH DEMONSTRATION")
    print("="*80)
    print("This demo shows the original tabu search algorithm with timing metrics")
    print("on various TSP problems from TSPLIB.")
    print()
    
    # Test problems with different characteristics
    test_problems = [
        ("berlin52", 200, 10, 50),    # Medium size, moderate iterations
        ("att48", 150, 15, 30),       # Different distance metric
        ("eil51", 100, 8, 25),        # Smaller problem, fewer iterations
    ]
    
    results = []
    
    for problem_name, iterations, tabu_size, max_no_improvement in test_problems:
        print(f"\n{'='*60}")
        print(f"TESTING PROBLEM: {problem_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load problem
            print(f"Loading TSP problem: {problem_name}")
            problem = load_tsp_problem(problem_name)
            print(f"Problem loaded: {problem.dimension} cities")
            
            # Create tabu search instance
            tabu_search = VanillaTabuSearch(
                problem=problem,
                tabu_size=tabu_size,
                max_iterations=iterations,
                max_no_improvement=max_no_improvement
            )
            
            print(f"\nParameters:")
            print(f"  - Max iterations: {iterations}")
            print(f"  - Tabu list size: {tabu_size}")
            print(f"  - Max no improvement: {max_no_improvement}")
            
            # Run tabu search
            print(f"\nStarting Vanilla Tabu Search...")
            start_time = time.perf_counter()
            
            best_solution, metrics = tabu_search.solve(verbose=False)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Add problem name to metrics
            metrics['problem_name'] = problem.name
            
            # Print results
            print_metrics(metrics)
            
            # Store results for summary
            results.append({
                'problem': problem_name,
                'dimension': problem.dimension,
                'initial_cost': metrics['initial_cost'],
                'final_cost': metrics['best_cost'],
                'improvement': metrics['initial_cost'] - metrics['best_cost'],
                'improvement_pct': ((metrics['initial_cost'] - metrics['best_cost']) / metrics['initial_cost'] * 100),
                'iterations': metrics['iterations'],
                'improvements': metrics['improvements'],
                'time': metrics['total_time'],
                'iterations_per_sec': metrics['iterations'] / metrics['total_time']
            })
            
        except Exception as e:
            print(f"Error testing {problem_name}: {e}")
            continue
    
    # Print summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL TESTS")
        print(f"{'='*80}")
        print(f"{'Problem':<12} {'Cities':<6} {'Initial':<10} {'Final':<10} {'Improve':<10} {'Improve%':<8} {'Time(s)':<8} {'Iter/sec':<8}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['problem']:<12} {result['dimension']:<6} "
                  f"{result['initial_cost']:<10.2f} {result['final_cost']:<10.2f} "
                  f"{result['improvement']:<10.2f} {result['improvement_pct']:<8.2f} "
                  f"{result['time']:<8.3f} {result['iterations_per_sec']:<8.1f}")
        
        # Calculate averages
        avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        avg_iter_per_sec = sum(r['iterations_per_sec'] for r in results) / len(results)
        
        print("-" * 80)
        print(f"{'AVERAGE':<12} {'':<6} {'':<10} {'':<10} {'':<10} "
              f"{avg_improvement:<8.2f} {avg_time:<8.3f} {avg_iter_per_sec:<8.1f}")
        
        print(f"\nKey Insights:")
        print(f"- Average improvement: {avg_improvement:.2f}%")
        print(f"- Average execution time: {avg_time:.3f} seconds")
        print(f"- Average iterations per second: {avg_iter_per_sec:.1f}")
        print(f"- Total problems tested: {len(results)}")
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("The vanilla tabu search implementation provides:")
    print("✓ Original tabu search algorithm with 2-opt neighborhood")
    print("✓ Comprehensive timing and performance metrics")
    print("✓ Support for different TSP problem formats (EUC_2D, ATT)")
    print("✓ Configurable parameters (iterations, tabu size, stopping criteria)")
    print("✓ Progress tracking and early stopping")
    print("✓ Detailed performance analysis and visualization capabilities")


if __name__ == "__main__":
    run_demo()
