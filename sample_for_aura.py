"""
sample_for_aura.py — Create a stratified sample of Amazon-Products.csv sized
to fit inside Neo4j Aura Free's limits (200,000 nodes / 400,000 relationships).

Run this ONCE, locally, before seeding Aura. Keep the original full CSV where
it is for local dev / your thesis — this only creates a smaller COPY for the
deployed instance.

    python sample_for_aura.py

Produces: data_aura/Amazon-Products.csv (a new folder, so seed.py's
CSV_FOLDER can point at just this one file without touching the original
data/ folder).
"""

import os
import pandas as pd

SOURCE_CSV = os.path.join("data", "Amazon-Products.csv")
OUTPUT_DIR = "data_aura"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Amazon-Products.csv")

# 150,000 products -> 150,000 nodes + ~112 category nodes + 1 brand node
# (~75% of Aura Free's 200k node cap), and 300,000 relationships
# (~75% of Aura Free's 400k relationship cap). Leaves ~25% headroom for
# live User/SearchQuery growth from actual app usage.
TARGET_ROWS = 150_000


def main():
    df = pd.read_csv(SOURCE_CSV)
    total = len(df)
    frac = TARGET_ROWS / total

    # Stratified by sub_category so every category survives proportionally —
    # a plain .head(TARGET_ROWS) would keep only the first few categories in
    # the file and silently drop the other ~100.
    sampled = (
        df.groupby("sub_category", group_keys=False)[df.columns.tolist()]
          .apply(lambda g: g.sample(frac=frac, random_state=42) if len(g) > 1 else g)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sampled.to_csv(OUTPUT_CSV, index=False)

    n = len(sampled)
    print(f"Original rows:              {total:,}")
    print(f"Sampled rows:               {n:,}")
    print(f"Categories preserved:       {sampled['sub_category'].nunique()} / {df['sub_category'].nunique()}")
    print(f"Product nodes:              {n:,}")
    print(f"Relationships (2 per row):  {n * 2:,}")
    print(f"Node budget used:           {n / 200_000:.0%} of Aura Free's 200,000 cap")
    print(f"Relationship budget used:   {n * 2 / 400_000:.0%} of Aura Free's 400,000 cap")
    print(f"\nWrote: {OUTPUT_CSV}")
    print("Point CSV_FOLDER=data_aura in your .env before running seed.py against Aura.")


if __name__ == "__main__":
    main()
