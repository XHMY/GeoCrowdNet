import pandas as pd

df = pd.read_csv("logs/results.txt", names=["exp_name", "seed", "test_accuracy"])
df = df.groupby("exp_name").mean()
print(df)