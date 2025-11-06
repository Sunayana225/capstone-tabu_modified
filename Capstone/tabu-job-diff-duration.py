import random
import matplotlib.pyplot as plt

# Jobs: id, duration, deadline, profit
jobs = [
    {"id": 1, "duration": 2, "deadline": 4, "profit": 50},
    {"id": 2, "duration": 1, "deadline": 2, "profit": 30},
    {"id": 3, "duration": 3, "deadline": 6, "profit": 70},
    {"id": 4, "duration": 2, "deadline": 5, "profit": 40},
]

# Evaluate a sequence
def evaluate(sequence, jobs):
    max_deadline = max(job["deadline"] for job in jobs)
    timeline = [None] * (max_deadline + 1)
    total_profit = 0
    schedule = []

    for job_id in sequence:
        job = next(j for j in jobs if j["id"] == job_id)
        d = job["duration"]

        # try to place job before its deadline
        for start in range(job["deadline"] - d + 1, 0, -1):  # latest possible start
            if all(timeline[t] is None for t in range(start, start + d)):
                for t in range(start, start + d):
                    timeline[t] = job
                total_profit += job["profit"]
                schedule.append((job["id"], start, start + d))
                break

    return total_profit, schedule

# Tabu search (same as before, just using new evaluate)
def tabu_search(jobs, iterations=100, tabu_tenure=5, neighbors=20):
    job_ids = [job["id"] for job in jobs]
    current = job_ids[:]
    random.shuffle(current)
    best = current[:]
    best_profit, _ = evaluate(best, jobs)
    tabu_list = {}

    for it in range(iterations):
        candidate = None
        candidate_profit = -1
        for _ in range(neighbors):
            i, j = random.sample(range(len(job_ids)), 2)
            neighbor = current[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            move = (min(neighbor[i], neighbor[j]), max(neighbor[i], neighbor[j]))
            if move in tabu_list and tabu_list[move] > it:
                continue
            profit, _ = evaluate(neighbor, jobs)
            if profit > candidate_profit:
                candidate, candidate_profit, candidate_move = neighbor, profit, move

        if candidate is not None:
            current = candidate
            if candidate_profit > best_profit:
                best, best_profit = candidate, candidate_profit
            tabu_list[candidate_move] = it + tabu_tenure
        print(f"Iter {it+1}: Current profit={candidate_profit}, Best profit={best_profit}")

    return best, best_profit, evaluate(best, jobs)[1]

# Visualization
def visualize(schedule, profit):
    fig, ax = plt.subplots(figsize=(8, 2))
    for job_id, start, end in schedule:
        ax.barh(0, end-start, left=start-1, edgecolor="black", color="skyblue")
        ax.text((start+end)/2 - 1, 0, f"J{job_id}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Time")
    ax.set_title(f"Job Schedule with Durations (Profit={profit})")
    plt.show()

# Run
best_seq, best_profit, schedule = tabu_search(jobs, iterations=50)
print("\nBest job sequence:", best_seq)
print("Best profit:", best_profit)
print("Schedule:", schedule)
visualize(schedule, best_profit)
