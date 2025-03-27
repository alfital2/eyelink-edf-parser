from edf_pipeline import run_pipeline
import pandas as pd

results = run_pipeline("185988598.edf")

results["fixations"]["timestamp"] = results["fixations"]["start_time"]
results["saccades"]["timestamp"] = results["saccades"]["start_time"]
results["blinks"]["timestamp"] = results["blinks"]["start_time"]
results["messages"]["timestamp"] = results["messages"]["timestamp"]

combined_df = pd.concat([
    results["fixations"],
    results["saccades"],
    results["blinks"],
    results["messages"]
], ignore_index=True)

combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

combined_df.to_csv("eyelink_combined.csv", index=False)

#added nothing