import pandas as pd

df = pd.read_csv("data/nfl_ats_model_dataset_with_players.csv")

print(df.shape)
print(df.columns[:25])
print(df.head(3).to_string())
