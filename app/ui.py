from __future__ import annotations

from typing import List, Optional

import requests
import os
import streamlit as st


API_BASE = "http://127.0.0.1:8000"


def plot_tour(coordinates_x: List[float], coordinates_y: List[float], path: List[int]) -> None:
	# Streamlit's built-in charting prefers dataframes, but we can draw via matplotlib implicitly using st.pyplot
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(figsize=(6, 6))
	# Close the tour by appending the first
	ordered_x = [coordinates_x[i] for i in path] + [coordinates_x[path[0]]]
	ordered_y = [coordinates_y[i] for i in path] + [coordinates_y[path[0]]]
	ax.plot(ordered_x, ordered_y, "-o", markersize=3)
	ax.set_title("TSP Tour")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.grid(True)
	st.pyplot(fig)


def _load_problem_presets() -> List[str]:
	try:
		import os
		from pathlib import Path
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
	except Exception:
		return ["0000", "berlin52", "a280", "eil76", "kroA100"]


def main() -> None:
	st.set_page_config(page_title="Tabu Search TSP", layout="wide")

	# Main navigation
	st.sidebar.title("🔍 Navigation")
	page = st.sidebar.radio(
		"Choose Page:",
		["🏠 Tabu Search TSP", "📊 Algorithm Comparison"]
	)
	
	if page == "📊 Algorithm Comparison":
		# Import and run comparison dashboard
		try:
			from .comparison_ui import main as comparison_main
			comparison_main(standalone=False)
			return
		except ImportError:
			from comparison_ui import main as comparison_main
			comparison_main(standalone=False)
			return

	# Original Tabu Search TSP page content
	with st.sidebar:
		st.markdown("---")
		st.markdown("**What is this?**")
		st.write(
			"This app runs a Tabu Search solver on TSPLIB problems. "
			"Tabu Search explores neighbor tours while using a short-term memory (tabu list) to avoid cycles "
			"and escape local minima."
		)
		presets = _load_problem_presets()
		default_idx = presets.index("berlin52") if "berlin52" in presets else 0
		selected_preset = st.selectbox("Choose a problem preset", presets, index=default_idx)
		st.caption("Files are resolved from `tabu-search/problems/tsp/<name>/<name>.tsp`.")

		# Optional Gemini explanation
		st.markdown("**Optional: Explanation**")
		use_gemini = st.checkbox("Generate a plain-English explanation (Gemini)")
		gemini_key: Optional[str] = None
		default_key = ""
		try:
			default_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))  # type: ignore[attr-defined]
		except Exception:
			default_key = os.getenv("GEMINI_API_KEY", "")
		if use_gemini:
			gemini_key = default_key or None
			if gemini_key:
				st.caption("Using configured API key from app secrets.")
			else:
				st.warning("No Gemini API key configured. Add it to .streamlit/secrets.toml as GEMINI_API_KEY.")

	st.title("Tabu Search for TSP")
	st.markdown("Use a preset or type a name, configure parameters, and click Solve.")

	with st.expander("How to use", expanded=False):
		st.markdown(
			"- Enter a TSPLIB problem name like `berlin52`, `a280`, `eil76`, `kroA100`.\n"
			"- The app looks for files in `tabu-search/problems/tsp/<name>/<name>.tsp`.\n"
			"- Increase **Iterations** for better solutions (slower).\n"
			"- **Tabu size** controls how many recent moves are forbidden (helps escape cycles).\n"
			"- **Shuffle after** triggers a shake-up after N non-improving steps.\n"
			"- **Pool size** is the parallel workers count (0 = auto)."
		)

	with st.form("solve_form"):
		problem = st.text_input(
			"Problem name",
			value=selected_preset,
			help="Type a TSPLIB problem (e.g., berlin52). We load it from problems/tsp/<name>/<name>.tsp",
			placeholder="berlin52",
		)
		st.caption("Examples: berlin52, a280, eil76, kroA100, pr1002 (depending on what exists in problems/tsp)")
		col1, col2, col3 = st.columns(3)
		with col1:
			iterations = st.number_input(
				"Iterations",
				min_value=1,
				value=500,
				help="More iterations usually improve solution quality but take longer",
			)
		with col2:
			tabu_size = st.number_input(
				"Tabu size",
				min_value=1,
				value=10,
				help="Length of tabu list (short-term memory) to prevent cycling",
			)
		with col3:
			shuffle_after = st.number_input(
				"Shuffle after",
				min_value=0,
				value=50,
				help="After this many non-improving iterations, randomly reshuffle the tour",
			)

		col4, col5 = st.columns(2)
		with col4:
			pool_size = st.number_input("Pool size (0=auto)", min_value=0, value=0)
		with col5:
			verbose = st.checkbox("Verbose progress")

		algorithm = st.radio(
			"Algorithm",
			options=["modified", "vanilla"],
			index=0,
			help="Compare the built-in modified tabu vs. a vanilla baseline",
			horizontal=True,
		)

		left, right = st.columns([1,1])
		with left:
			submitted = st.form_submit_button("Solve", use_container_width=True)
		with right:
			compare_both = st.form_submit_button("Compare both", use_container_width=True)
		st.caption("Note: 'Compare both' runs modified and vanilla regardless of the radio selection.")

	if submitted or compare_both:
		problem_name = problem if problem else selected_preset
		if not problem_name or not problem_name.strip():
			st.error("Please enter a problem name, e.g., berlin52.")
			return
		if compare_both:
			# Run modified then vanilla
			with st.spinner("Running modified and vanilla..."):
				payload_mod = {
					"problem": problem_name,
					"iterations": int(iterations),
					"shuffle_after": int(shuffle_after),
					"tabu_size": int(tabu_size),
					"pool_size": int(pool_size) if int(pool_size) > 0 else None,
					"verbose": bool(verbose),
					"algorithm": "modified",
				}
				payload_van = dict(payload_mod)
				payload_van["algorithm"] = "vanilla"
				# Modified
				try:
					resp_mod = requests.post(f"{API_BASE}/solve", json=payload_mod, timeout=1800)
					resp_van = requests.post(f"{API_BASE}/solve", json=payload_van, timeout=1800)
				except requests.exceptions.RequestException as e:
					st.error(f"Connection to API failed: {e}. Ensure the API is running with .\\run_api.ps1")
					return
				if resp_mod.status_code != 200:
					st.error(f"Modified failed: {resp_mod.status_code} - {resp_mod.text}")
					return
				if resp_van.status_code != 200:
					st.error(f"Vanilla failed: {resp_van.status_code} - {resp_van.text}")
					return
				data_mod = resp_mod.json()["data"]
				data_van = resp_van.json()["data"]

			# Comparison metrics
			cm1, cm2, cm3 = st.columns(3)
			cm1.metric("Modified cost", f"{data_mod['solution']['cost']:.2f}")
			cm2.metric("Vanilla cost", f"{data_van['solution']['cost']:.2f}")
			delta_cost = data_van['solution']['cost'] - data_mod['solution']['cost']
			pct = (delta_cost / data_van['solution']['cost'] * 100.0) if data_van['solution']['cost'] else 0.0
			cm3.metric("Improvement", f"{delta_cost:.2f}", f"{pct:.1f}%")

			cm4, cm5 = st.columns(2)
			ms_mod = data_mod.get('elapsed_ms')
			ms_van = data_van.get('elapsed_ms')
			cm4.metric("Modified time (ms)", f"{ms_mod:.1f}" if isinstance(ms_mod, (int, float)) else "N/A")
			cm5.metric("Vanilla time (ms)", f"{ms_van:.1f}" if isinstance(ms_van, (int, float)) else "N/A")

			st.subheader("Modified tour")
			plot_tour(data_mod["coordinates"]["x"], data_mod["coordinates"]["y"], data_mod["solution"]["path"])
			st.subheader("Vanilla tour")
			plot_tour(data_van["coordinates"]["x"], data_van["coordinates"]["y"], data_van["solution"]["path"])

			# Optional explanation for modified winner
			winner = data_mod if data_mod['solution']['cost'] <= data_van['solution']['cost'] else data_van
			winner_label = "modified" if winner is data_mod else "vanilla"
			if use_gemini and gemini_key:
				try:
					from app.gemini_helper import generate_explanation  # type: ignore
				except Exception:
					from gemini_helper import generate_explanation  # type: ignore
				with st.spinner("Generating explanation for best result..."):
					try:
						explanation = generate_explanation(gemini_key, winner)
					except Exception as ex:  # noqa: BLE001
						explanation = f"Failed to generate explanation: {ex}"
				st.subheader(f"Explanation ({winner_label})")
				st.write(explanation)
			return

		with st.spinner("Running tabu search..."):
			payload = {
				"problem": problem_name,
				"iterations": int(iterations),
				"shuffle_after": int(shuffle_after),
				"tabu_size": int(tabu_size),
				"pool_size": int(pool_size) if int(pool_size) > 0 else None,
				"verbose": bool(verbose),
				"algorithm": algorithm,
			}
			resp = requests.post(f"{API_BASE}/solve", json=payload, timeout=1200)
			if resp.status_code != 200:
				st.error(f"Error: {resp.status_code} - {resp.text}")
				return
			data = resp.json()["data"]

		# Result header card
		c1, c2, c3, c4 = st.columns(4)
		algo_label = "Modified Tabu" if algorithm == "modified" else "Vanilla Tabu"
		c1.metric("Problem", f"{data['problem']} ({algo_label})")
		c2.metric("Cities", data["dimension"])
		c3.metric("Cost", f"{data['solution']['cost']:.2f}")
		ms_single = data.get('elapsed_ms')
		c4.metric("Elapsed (ms)", f"{ms_single:.1f}" if isinstance(ms_single, (int, float)) else "N/A")

		# Tour plot
		coords_x = data["coordinates"]["x"]
		coords_y = data["coordinates"]["y"]
		path = data["solution"]["path"]
		if coords_x and coords_y and path:
			st.subheader("TSP Tour")
			plot_tour(coords_x, coords_y, path)

		# Optional Gemini explanation
		if use_gemini and gemini_key:
			try:
				from app.gemini_helper import generate_explanation  # type: ignore
			except Exception:
				from gemini_helper import generate_explanation  # type: ignore
			with st.spinner("Generating explanation..."):
				try:
					explanation = generate_explanation(gemini_key, data)
				except Exception as ex:  # noqa: BLE001
					explanation = f"Failed to generate explanation: {ex}"
			st.subheader("Explanation")
			st.write(explanation)

		# Differences and metrics help
		with st.expander("How modified vs vanilla differ"):
			st.markdown(
				"- Modified: Uses multiple neighborhoods and shuffle-after strategy; can parallelize with a pool.\n"
				"- Vanilla: Single-neighborhood baseline with a 'max_no_improvement' stop.\n"
				"- Metrics: Lower cost is better; elapsed is wall-clock time."
			)


if __name__ == "__main__":
	main()


