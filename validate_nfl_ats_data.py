#!/usr/bin/env python3
import pandas as pd
from ydata_profiling import ProfileReport

# Load your enriched dataset
df = pd.read_csv("data/nfl_ats_model_dataset_with_context.csv")

# Generate validation report
profile = ProfileReport(
    df,
    title="ATS Dataset Validation Report",
    explorative=True
)

# Save to HTML
profile.to_file("validation_report.html")
print("✅ Report saved as validation_report.html")
