from pathlib import Path
import json



PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACKING_DIR = PROJECT_ROOT / "tracking"
LEADERBOARD_PATH = TRACKING_DIR / "leaderboard.json"



def load_leaderboard(top_k=2, metric="val_auprc_mean"):

    # ensure folder exists
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # if file doesn't exist → create default
    if not LEADERBOARD_PATH.exists():
        default = {
            "metric_name": metric,
            "top_k": top_k,
            "runs": []
        }
        LEADERBOARD_PATH.write_text(json.dumps(default, indent=2))
        return default

    return json.loads(LEADERBOARD_PATH.read_text())


# see later if i use it
def save_leaderboard(board):
    LEADERBOARD_PATH.write_text(json.dumps(board, indent=2))


# update leaderboard
def update_leaderboard(run_id: str, score: float):
    board = load_leaderboard()

    metric_name = board["metric_name"]
    top_k = board["top_k"]

    old_runs = board["runs"].copy()

    # 1. Insert new run
    board["runs"].append({
        "run_id": run_id,
        "score": float(score)
    })

    # 2. Sort descending
    board["runs"] = sorted(
        board["runs"],
        key=lambda x: x["score"],
        reverse=True
    )

    # 3. Trim to top_k
    board["runs"] = board["runs"][:top_k]

    # 4. Save
    save_leaderboard(board)

    # 5. Compute info
    new_ids = {r["run_id"] for r in board["runs"]}
    old_ids = {r["run_id"] for r in old_runs}

    removed_ids = old_ids - new_ids

    rank = None
    for i, r in enumerate(board["runs"], start=1):
        if r["run_id"] == run_id:
            rank = i
            break

    return {
        "is_top_k": rank is not None,
        "rank": rank,                 # 1,2,... or None
        "removed_run_ids": list(removed_ids),
        "metric_name": metric_name
    }



#---------------Get best run ids--------------------------#
def get_best_run_id():
    lb = load_leaderboard()
    if not lb or not lb["runs"]:
        return None
    return lb["runs"][0]["run_id"]



def get_top_k_run_ids(k=None):
    lb = load_leaderboard()
    if not lb:
        return []
    runs = lb["runs"]
    if k:
        runs = runs[:k]
    return [r["run_id"] for r in runs]
