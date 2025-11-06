from __future__ import annotations

from typing import Any, Dict, Optional


def generate_explanation(api_key: str, result: Dict[str, Any]) -> str:
	"""Generate a short human-friendly explanation using Gemini, if SDK is available.

	This imports the SDK lazily to keep it optional for users who don't need it.
	"""
	try:
		import google.generativeai as genai  # type: ignore
	except Exception:
		return (
			"Gemini SDK not installed. Run: pip install google-generativeai and try again."
		)

	genai.configure(api_key=api_key)
	model = genai.GenerativeModel("gemini-2.0-flash")

	problem = result.get("problem")
	dimension = result.get("dimension")
	cost = result.get("solution", {}).get("cost")
	params = result.get("parameters", {})
	iterations = params.get("iterations")
	tabu_size = params.get("tabu_size")
	shuffle_after = params.get("shuffle_after")
	pool_size = params.get("pool_size")

	prompt = (
		"Explain in 4-6 sentences, for a general technical audience, what a Tabu Search "
		"solver just did on a TSPLIB instance. Include how tabu list and iterations help, "
		"and summarize the result succinctly. Avoid code.\n\n"
		f"Problem: {problem} (cities: {dimension})\n"
		f"Parameters: iterations={iterations}, tabu_size={tabu_size}, shuffle_after={shuffle_after}, pool_size={pool_size}\n"
		f"Result cost: {cost}"
	)

	response = model.generate_content(prompt)
	text = getattr(response, "text", None) or ""
	return text.strip()


