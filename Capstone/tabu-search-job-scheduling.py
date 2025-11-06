import random
import matplotlib.pyplot as plt

# =============================
# Job Data
# =============================
jobs = [
    {"id": 1, "deadline": 2, "profit": 100},
    {"id": 2, "deadline": 1, "profit": 19},
    {"id": 3, "deadline": 2, "profit": 27},
    {"id": 4, "deadline": 1, "profit": 25},
    {"id": 5, "deadline": 3, "profit": 15},
]

# =============================
# Evaluate a sequence
# =============================
def evaluate(sequence, jobs):
    """Return profit and slots assignment for given job order"""
    max_deadline = max(job["deadline"] for job in jobs)
    slots = [None] * (max_deadline + 1)  # 1..max_deadline
    total_profit = 0

    for job_id in sequence:
        job = next(j for j in jobs if j["id"] == job_id)
        for t in range(job["deadline"], 0, -1):
            if slots[t] is None:
                slots[t] = job
                total_profit += job["profit"]
                break

    return total_profit, slots

# =============================
# Tabu Search
# =============================
def tabu_search(jobs, iterations=100, tabu_tenure=5, neighbors=20):
    job_ids = [job["id"] for job in jobs]

    # Start with random order
    current = job_ids[:]
    random.shuffle(current)
    best = current[:]
    best_profit, _ = evaluate(best, jobs)

    # Tabu list stores forbidden swaps
    tabu_list = {}

    for it in range(iterations):
        candidate = None
        candidate_profit = -1

        # Explore neighbors
        for _ in range(neighbors):
            i, j = random.sample(range(len(job_ids)), 2)
            neighbor = current[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            move = (min(neighbor[i], neighbor[j]), max(neighbor[i], neighbor[j]))

            if move in tabu_list and tabu_list[move] > it:
                continue  # tabu move

            profit, _ = evaluate(neighbor, jobs)

            if profit > candidate_profit:
                candidate, candidate_profit, candidate_move = neighbor, profit, move

        # Update current solution
        if candidate is not None:
            current = candidate
            if candidate_profit > best_profit:
                best, best_profit = candidate, candidate_profit
            tabu_list[candidate_move] = it + tabu_tenure

        print(f"Iter {it+1}: Current profit={candidate_profit}, Best profit={best_profit}")

    return best, best_profit

# =============================
# Visualization (Timeline)
# =============================
def visualize_schedule(best_sequence, jobs):
    profit, slots = evaluate(best_sequence, jobs)
    fig, ax = plt.subplots(figsize=(8, 2))

    for t in range(1, len(slots)):
        if slots[t] is not None:
            job_id = slots[t]["id"]
            profit_val = slots[t]["profit"]
            ax.barh(0, 1, left=t-1, edgecolor="black", color="skyblue")
            ax.text(t-0.5, 0, f"J{job_id}\nP{profit_val}", 
                    ha="center", va="center", fontsize=9)

    ax.set_xlim(0, len(slots)-1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(range(len(slots)))
    ax.set_yticks([])
    ax.set_xlabel("Time Slots")
    ax.set_title(f"Job Schedule (Profit={profit})")
    plt.show()

# =============================
# Run Tabu Search + Visualize
# =============================
if __name__ == "__main__":
    best_seq, best_profit = tabu_search(jobs, iterations=50)
    print("\nBest job sequence:", best_seq)
    print("Best profit:", best_profit)
    visualize_schedule(best_seq, jobs)
