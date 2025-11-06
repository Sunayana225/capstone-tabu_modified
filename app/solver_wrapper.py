from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _ensure_tabu_package_loaded() -> Path:

	current_dir = Path(__file__).resolve().parent
	root_dir = current_dir.parent
	tabu_dir = root_dir / "tabu-search"
	if str(tabu_dir) not in sys.path:
		sys.path.insert(0, str(tabu_dir))
	return tabu_dir


def run_tsp(
	problem: str,
	iterations: int = 500,
	shuffle_after: int = 50,
	tabu_size: int = 10,
	pool_size: int | None = None,
	verbose: bool = False,
) -> Dict[str, Any]:
    """Run the bundled TSP Tabu Search (modified) and return structured results."""

    return _run_tsp_modified(
        problem=problem,
        iterations=iterations,
        shuffle_after=shuffle_after,
        tabu_size=tabu_size,
        pool_size=pool_size,
        verbose=verbose,
    )


def _run_tsp_modified(
	problem: str,
	iterations: int = 500,
	shuffle_after: int = 50,
	tabu_size: int = 10,
	pool_size: int | None = None,
	verbose: bool = False,
) -> Dict[str, Any]:
	"""Modified implementation in ts.tsp, mirrors `tabu-search/tsp.py`."""
	tabu_dir = _ensure_tabu_package_loaded()

	# Import after sys.path is prepared
	from ts import tsp as tsp_mod  # type: ignore

	# The algorithm reads problems via relative paths under working dir.
	# Temporarily chdir into the tabu-search folder for correct file resolution.
	previous_cwd = Path.cwd()
	os.chdir(tabu_dir)
	try:
		start_ns = time.perf_counter_ns()
		# Load problem
		tsp_mod.TSPPathSolution.import_problem(problem)

		# Configure tabu lists
		tsp_mod.Swap.reset_tabu(maxlen=tabu_size)
		tsp_mod.SegmentShift.reset_tabu(maxlen=tabu_size)
		tsp_mod.SegmentReverse.reset_tabu(maxlen=tabu_size)

		# Run search
		actual_pool = pool_size if pool_size and pool_size > 0 else (os.cpu_count() or 1)
		solution = tsp_mod.TSPPathSolution.tabu_search(
			pool_size=actual_pool,
			iterations_count=iterations,
			use_tqdm=verbose,
			shuffle_after=shuffle_after,
		)

		# Validate
		check = tsp_mod.TSPPathSolution(after=solution.after, before=solution.before)
		assert check.cost() == solution.cost()

		elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

		# Collect coordinates for plotting in UI
		x = list(getattr(tsp_mod.TSPPathSolution, "x", ()))
		y = list(getattr(tsp_mod.TSPPathSolution, "y", ()))

		result: Dict[str, Any] = {
			"problem": getattr(tsp_mod.TSPPathSolution, "problem_name", problem),
			"dimension": getattr(tsp_mod.TSPPathSolution, "dimension", len(x)),
			"parameters": {
				"iterations": iterations,
				"tabu_size": tabu_size,
				"shuffle_after": shuffle_after,
				"pool_size": actual_pool,
			},
			"solution": {
				"path": list(solution.path),
				"cost": float(solution.cost()),
			},
			"coordinates": {"x": x, "y": y},
			"elapsed_ms": elapsed_ms,
		}
		return result
	finally:
		os.chdir(previous_cwd)


def run_tsp_vanilla(
	problem: str,
	iterations: int = 1000,
	tabu_size: int = 10,
	max_no_improvement: int = 100,
	verbose: bool = False,
) -> Dict[str, Any]:
	"""Run the vanilla Tabu Search implementation in `tabu-search/vanilla_tabu.py`."""
	root_dir = Path(__file__).resolve().parent.parent
	tabu_dir = _ensure_tabu_package_loaded()

	# Import after sys.path prepared
	previous_cwd = Path.cwd()
	os.chdir(tabu_dir)
	try:
		from vanilla_tabu import load_tsp_problem, VanillaTabuSearch  # type: ignore

		problem_obj = load_tsp_problem(problem)
		import time
		start_ns = time.perf_counter_ns()
		tabu_search = VanillaTabuSearch(
			problem=problem_obj,
			tabu_size=tabu_size,
			max_iterations=iterations,
			max_no_improvement=max_no_improvement,
		)
		best_solution, metrics = tabu_search.solve(verbose=verbose)
		elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

		coords = list(problem_obj.coordinates)
		x = [c[0] for c in coords]
		y = [c[1] for c in coords]

		return {
			"problem": problem_obj.name,
			"dimension": problem_obj.dimension,
			"parameters": {
				"iterations": iterations,
				"tabu_size": tabu_size,
				"max_no_improvement": max_no_improvement,
			},
			"solution": {
				"path": list(best_solution.path),
				"cost": float(best_solution.cost()),
			},
			"coordinates": {"x": x, "y": y},
			"metrics": metrics,
			"elapsed_ms": elapsed_ms,
		}
	finally:
		os.chdir(previous_cwd)


def run_tsp_dump_json(output_path: str | Path, **kwargs: Any) -> Path:
	"""Convenience to run and dump results to a JSON file."""
	res = run_tsp(**kwargs)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(res, indent=2))
	return output_path


