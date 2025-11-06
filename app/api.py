from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from .solver_wrapper import run_tsp, run_tsp_vanilla
from .capstone_wrapper import (
    run_tsp_genetic_algorithm, run_tsp_ant_colony_optimization, 
    run_tsp_simulated_annealing, run_job_scheduling_tabu_search,
    run_tsp_algorithm_comparison, run_all_algorithms_comparison,
    _get_available_tsp_problems, _get_available_d2d_problems, 
    _get_all_available_problems, run_d2d_tabu_search
)


class SolveRequest(BaseModel):
	problem: str = Field(..., description="Problem name under problems/tsp, e.g. 'berlin52' or 'berlin52.tsp'")
	iterations: int = Field(500, ge=1)
	shuffle_after: int = Field(50, ge=0)
	tabu_size: int = Field(10, ge=1)
	pool_size: int | None = Field(None, ge=1)
	verbose: bool = False
	algorithm: str = Field("modified", description="'modified' or 'vanilla'")


class TSPAlgorithmRequest(BaseModel):
	algorithm: str = Field(..., description="'genetic_algorithm', 'ant_colony_optimization', or 'simulated_annealing'")
	problem: str = Field("berlin52", description="TSP problem name")
	verbose: bool = False
	# Algorithm-specific parameters
	population_size: Optional[int] = Field(100, description="GA: Population size")
	generations: Optional[int] = Field(200, description="GA: Number of generations")
	mutation_rate: Optional[float] = Field(0.02, description="GA: Mutation rate")
	n_ants: Optional[int] = Field(30, description="ACO: Number of ants")
	n_iterations: Optional[int] = Field(150, description="ACO: Number of iterations")
	decay: Optional[float] = Field(0.95, description="ACO: Pheromone decay")
	temperature: Optional[float] = Field(100, description="SA: Initial temperature")
	alpha: Optional[float] = Field(0.995, description="SA: Cooling rate")
	stopping_iter: Optional[int] = Field(5000, description="SA: Stopping iterations")
	runs: Optional[int] = Field(1, description="Number of independent runs")


class JobSchedulingRequest(BaseModel):
	job_type: str = Field("basic", description="'basic' or 'duration'")
	iterations: int = Field(50, ge=1)
	tabu_tenure: int = Field(5, ge=1)
	neighbors: int = Field(20, ge=1)
	verbose: bool = False


class D2DRequest(BaseModel):
	problem: str = Field(..., description="D2D problem name (e.g., '6.5.1' for 6.5.1.txt)")
	iterations: int = Field(1000, ge=1)
	shuffle_after: int = Field(100, ge=0)
	tabu_size: int = Field(10, ge=1)
	verbose: bool = False


class ComparisonRequest(BaseModel):
	algorithms: Optional[List[str]] = Field(None, description="List of algorithms to compare")
	problem: str = Field("berlin52", description="Problem name")
	verbose: bool = False
	include_tabu: bool = Field(True, description="Include Tabu Search algorithms in comparison")


class SolveResponse(BaseModel):
	data: Dict[str, Any]


app = FastAPI(title="Tabu Search TSP API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.get("/available-problems")
def get_available_problems() -> Dict[str, Any]:
	"""Get list of available problems (TSP and D2D)."""
	try:
		all_problems = _get_all_available_problems()
		tsp_problems = all_problems["tsp"]
		d2d_problems = all_problems["d2d"]
		
		return {
			"problems": tsp_problems,  # For backward compatibility
			"tsp_problems": tsp_problems,
			"d2d_problems": d2d_problems,
			"total_tsp": len(tsp_problems),
			"total_d2d": len(d2d_problems)
		}
	except Exception as e:
		return {
			"problems": ["eil51", "berlin52", "gr17"], 
			"tsp_problems": ["eil51", "berlin52", "gr17"],
			"d2d_problems": [],
			"error": str(e)
		}


@app.post("/solve-d2d", response_model=SolveResponse)
def solve_d2d(req: D2DRequest) -> SolveResponse:
	"""Run Tabu Search on D2D (Door-to-Door) problem."""
	try:
		data = run_d2d_tabu_search(
			problem=req.problem,
			iterations=req.iterations,
			shuffle_after=req.shuffle_after,
			tabu_size=req.tabu_size,
			verbose=req.verbose
		)
		return SolveResponse(data=data)
	except FileNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve-tsp-algorithm", response_model=SolveResponse)
def solve_tsp_algorithm(req: TSPAlgorithmRequest) -> SolveResponse:
	"""Run a specific TSP algorithm from Capstone folder."""
	try:
		if req.algorithm == "genetic_algorithm":
			data = run_tsp_genetic_algorithm(
				problem=req.problem,
				population_size=req.population_size or 100,
				generations=req.generations or 200,
				mutation_rate=req.mutation_rate or 0.02,
				runs=req.runs or 1,
				verbose=req.verbose
			)
		elif req.algorithm == "ant_colony_optimization":
			data = run_tsp_ant_colony_optimization(
				problem=req.problem,
				n_ants=req.n_ants or 30,
				n_iterations=req.n_iterations or 150,
				decay=req.decay or 0.95,
				verbose=req.verbose
			)
		elif req.algorithm == "simulated_annealing":
			data = run_tsp_simulated_annealing(
				problem=req.problem,
				T=req.temperature or 100,
				alpha=req.alpha or 0.995,
				stopping_iter=req.stopping_iter or 5000,
				runs=req.runs or 1,
				verbose=req.verbose
			)
		else:
			raise HTTPException(status_code=400, detail=f"Unknown algorithm: {req.algorithm}")
		
		return SolveResponse(data=data)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve-job-scheduling", response_model=SolveResponse)
def solve_job_scheduling(req: JobSchedulingRequest) -> SolveResponse:
	"""Run Job Scheduling Tabu Search."""
	try:
		data = run_job_scheduling_tabu_search(
			job_type=req.job_type,
			iterations=req.iterations,
			tabu_tenure=req.tabu_tenure,
			neighbors=req.neighbors,
			verbose=req.verbose
		)
		return SolveResponse(data=data)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-tsp-algorithms", response_model=SolveResponse)
def compare_tsp_algorithms(req: ComparisonRequest) -> SolveResponse:
	"""Compare multiple TSP algorithms."""
	try:
		data = run_tsp_algorithm_comparison(
			algorithms=req.algorithms or ["genetic_algorithm", "ant_colony_optimization", "simulated_annealing"],
			problem=req.problem,
			verbose=req.verbose,
			include_tabu=req.include_tabu
		)
		return SolveResponse(data=data)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-all-algorithms", response_model=SolveResponse)
def compare_all_algorithms(verbose: bool = False) -> SolveResponse:
	"""Compare all available algorithms from Capstone folder."""
	try:
		data = run_all_algorithms_comparison(verbose=verbose)
		return SolveResponse(data=data)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
	try:
		if req.algorithm == "vanilla":
			data = run_tsp_vanilla(
				problem=req.problem,
				iterations=req.iterations,
				tabu_size=req.tabu_size,
				max_no_improvement=max(req.shuffle_after, 1),
				verbose=req.verbose,
			)
		else:
			data = run_tsp(
				problem=req.problem,
				iterations=req.iterations,
				shuffle_after=req.shuffle_after,
				tabu_size=req.tabu_size,
				pool_size=req.pool_size,
				verbose=req.verbose,
			)
		return SolveResponse(data=data)
	except FileNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(e))


