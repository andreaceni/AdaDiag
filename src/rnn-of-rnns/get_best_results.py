import os
import json
import pandas as pd

import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model (adadiag scn lstm rnn)")
parser.add_argument("--dataset", type=str, required=True, help="Dataset (smnist psmnist fordb etc)")
parser.add_argument("--test_score", type=float, required=True, help="Test score threshold (e.g. 0.92 for test_acc, 13.42 for test_RMSE)")
parser.add_argument("--RMSE", action="store_true", help="Inform to check for test_RMSE instead of test_acc")
parser.add_argument("--batch", type=str, default="128", help="Training batch size (e.g. 512)")
parser.add_argument("--JSON", action="store_true", help="If set, the full JSON line will be printed")
parser.add_argument("--mergeID", action="store_true", help="Keep only one best row per trial_id")
parser.add_argument("--block_config", type=str, default="3", help="Block configuration (e.g. 3 or 13)")
parser.add_argument("--folder", type=str, required=True, help="Folder containing the results (final_search, bayesian_search)")
args = parser.parse_args()

# Usage example:
# python get_best_results.py --mergeID --batch 512 --model adadiag --dataset smnist --test_score 0.972 --folder final_search
# python get_best_results.py --mergeID --RMSE --batch 1024 --model adadiag --dataset newstitlesentiment --test_score 14.1 --folder final_search

if args.model == "adadiag":
    model = "rnnassembly"
elif args.model == "scn":
    model = "scr"
else:
    model = args.model
dataset = args.dataset

batch = args.batch
block_config = args.block_config

# Change this to your folder path
folder = args.folder
if folder == 'bayesian_search':
    model = 'enhanced_rnnassembly'
#root_folder = f"/storagenfs/a052721/assemblyRNNs_har2/final_search/{dataset}/{model}"
root_folder = Path(f"/storagenfs/a052721/AdaDiag/{folder}/{dataset}/{model}")
if not root_folder.exists():
    print(f"❌ Path does not exist: {root_folder}")
    sys.exit(1)

# Change this to your desired test_acc threshold
test_score_threshold = args.test_score

results = []

print("Scanning folder:", root_folder)

for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith(".json"):  # adjust if needed
            filepath = os.path.join(dirpath, filename)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if not isinstance(data, dict):
                                continue  # skip anything that's not a dict

                            train_batch_size = data.get("config", {}).get("train_batch_size", None)
                            block_cfg = str(data.get("config", {}).get("block_config", None))

                            if args.RMSE:
                                test_score = data.get("test_RMSE", float('inf'))
                                if test_score <= test_score_threshold and str(train_batch_size) == batch:
                                    if model in ["rnnassembly", "scr"]:
                                        if block_cfg == block_config:
                                            results.append({
                                                "file": filepath,
                                                "trial_id": data.get("trial_id"),
                                                "training_iteration": data.get("training_iteration"),
                                                "test_RMSE": test_score,
                                                "train_RMSE": data.get("train_RMSE"),
                                                "train_loss": data.get("train_loss"),
                                                "test_loss": data.get("test_loss"),
                                                "batch_size": train_batch_size,
                                                "lr": data.get("lr"),
                                                "train_params": data.get("n_trainable_params"),
                                                "date": data.get("date"),
                                                "json_line": line,   # store full JSON line
                                            })
                                    else:
                                        # case of lstm or rnn
                                        results.append({
                                            "file": filepath,
                                            "trial_id": data.get("trial_id"),
                                            "training_iteration": data.get("training_iteration"),
                                            "test_RMSE": test_score,
                                            "train_RMSE": data.get("train_RMSE"),
                                            "train_loss": data.get("train_loss"),
                                            "test_loss": data.get("test_loss"),
                                            "batch_size": train_batch_size,
                                            "lr": data.get("lr"),
                                            "train_params": data.get("n_trainable_params"),
                                            "date": data.get("date"),
                                            "json_line": line,   # store full JSON line
                                        })
                            else:
                                test_score = data.get("test_acc", 0)
                                if test_score >= test_score_threshold and str(train_batch_size) == batch:
                                    if model in ["rnnassembly", "scr"]:
                                        if block_cfg == block_config:
                                            results.append({
                                                "file": filepath,
                                                "trial_id": data.get("trial_id"),
                                                "training_iteration": data.get("training_iteration"),
                                                "test_acc": test_score,
                                                "train_acc": data.get("train_acc"),
                                                "train_loss": data.get("train_loss"),
                                                "test_loss": data.get("test_loss"),
                                                "batch_size": train_batch_size,
                                                "lr": data.get("lr"),
                                                "train_params": data.get("n_trainable_params"),
                                                "date": data.get("date"),
                                                "json_line": line,   # store full JSON line
                                            })
                                    else:
                                        # case of lstm or rnn
                                        results.append({
                                            "file": filepath,
                                            "trial_id": data.get("trial_id"),
                                            "training_iteration": data.get("training_iteration"),
                                            "test_acc": test_score,
                                            "train_acc": data.get("train_acc"),
                                            "train_loss": data.get("train_loss"),
                                            "test_loss": data.get("test_loss"),
                                            "batch_size": train_batch_size,
                                            "lr": data.get("lr"),
                                            "train_params": data.get("n_trainable_params"),
                                            "date": data.get("date"),
                                            "json_line": line,   # store full JSON line
                                        })
                        except json.JSONDecodeError:
                            continue  # skip invalid JSON lines
            except Exception as e:
                print(f"⚠️ Skipping {filepath}: {e}")

print(f"\nFound {len(results)} entries matching criteria.")
# Convert to DataFrame
df = pd.DataFrame(results)
if not df.empty:
    if args.mergeID:
        if args.RMSE:
            # keep only the row with minimum test_RMSE per trial_id
            best_per_trial = (
                df.sort_values(["test_RMSE", "training_iteration"], ascending=[True, False])
                .groupby("trial_id")
                .head(1)
                .reset_index(drop=True)
            )
        else:
            # keep only the row with maximum test_acc per trial_id
            best_per_trial = (
                df.sort_values(["test_acc", "training_iteration"], ascending=[False, False])
                .groupby("trial_id")
                .head(1)
                .reset_index(drop=True)
            )
    else:
        if args.RMSE:
            best_per_trial = (
                df.sort_values(["test_RMSE", "training_iteration"], ascending=[True, False])
                .groupby("trial_id")
                .head(1)
                .reset_index(drop=True)
            )
        else:
            best_per_trial = (
                df.sort_values(["test_acc", "training_iteration"], ascending=[False, False])
                .groupby("trial_id")
                .head(1)
                .reset_index(drop=True)
            )

    print(f"\nAfter merging by ID trial, found {len(best_per_trial)} entries matching criteria.")

    print(best_per_trial.iloc[:, :-1] )  # print without the json_line column

    # Save results
    best_per_trial.to_csv("best_filtered_results.csv", index=False)
    print("\n✅ Saved best results to best_filtered_results.csv")

    if args.JSON:
        # print the full original JSON line exactly as it appeared in the file
        print("\n✅ Best JSON lines per trial:")
        for idx, row in best_per_trial.iterrows():
            print(f"\n--- Trial {row['trial_id']} ---")
            print(row["json_line"])

else:
    print("⚠️ No matching entries found.")