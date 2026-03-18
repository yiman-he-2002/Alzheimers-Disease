import pandas as pd

df = pd.read_csv("image_multi_modal.csv")

df_unique = df.drop_duplicates(subset=["PTID", "DIAGNOSIS"])

sampled_ids = (
    df_unique.groupby("DIAGNOSIS")["PTID"]
    .apply(lambda x: x.sample(n=min(25, len(x)), random_state=42))
)

for diag, ids in sampled_ids.groupby(level=0):
    print(f"Diagnosis {diag}:")
    print(list(ids.values))   
    print("-" * 50)
    
for diag, ids in sampled_ids.groupby(level=0):
    print(", ".join(ids))