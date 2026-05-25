# ============================================================
#  brand_analysis.py  —  Pre-computed Indian brand analysis
#  Run this ONCE after training to generate brand_data.json
#  python3 brand_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from scorer import (
    engineer_features, health_score_0_to_100,
    nutriscore_to_grade, compute_deception_score,
    FEATURE_COLS,
)

DATA_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
OUT_FILE = "brand_data.json"

# Well-known Indian brands to analyse
INDIAN_BRANDS = [
    "maggi", "parle", "britannia", "haldiram", "lays", "kurkure",
    "horlicks", "bournvita", "complan", "boost", "milo",
    "sunfeast", "hide and seek", "good day", "marie",
    "act ii", "bikaji", "bingo", "uncle chipps",
    "nestle", "cadbury", "kit kat", "dairy milk",
    "tropicana", "real", "minute maid", "paper boat",
    "patanjali", "aashirvaad", "fortune", "saffola",
    "munch", "kit kat", "5 star", "perk",
]

FEATURE_COLS_NEEDED = FEATURE_COLS + ["product_name", "brands",
                                       "categories", "nutriscore_score"]


def run_analysis():
    print("Loading dataset...")
    try:
        df = pd.read_csv(
            DATA_URL,
            usecols=lambda c: c in FEATURE_COLS_NEEDED,
            nrows=300_000,
            low_memory=False,
            compression="gzip",
            sep="\t",
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(df):,} rows")

    # Clean
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["brands"] = df.get("brands", pd.Series()).fillna("").str.lower()
    df["product_name"] = df.get("product_name", pd.Series()).fillna("")

    # Load model
    try:
        model  = joblib.load("food_scorer.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Model not found. Run colab_training.py first.")
        return

    # Score all products
    print("Scoring products...")
    results = []
    for _, row in df.iterrows():
        nutrients = {col: row.get(col, 0) for col in FEATURE_COLS}
        try:
            feat      = engineer_features(nutrients)
            feat_sc   = scaler.transform(feat)
            raw       = float(model.predict(feat_sc)[0])
            h_score   = health_score_0_to_100(raw)
            grade     = nutriscore_to_grade(raw)
            deception = compute_deception_score(
                str(row.get("product_name","")),
                "",
                h_score,
            )
            results.append({
                "product":    str(row.get("product_name",""))[:60],
                "brand":      str(row.get("brands",""))[:30],
                "health":     h_score,
                "grade":      grade,
                "deception":  deception["score"],
                "d_level":    deception["level"],
                "claims":     deception["claims"],
            })
        except Exception:
            continue

    df_scored = pd.DataFrame(results)
    print(f"Scored {len(df_scored):,} products")

    # ── BRAND LEVEL ANALYSIS ──────────────────────────────
    brand_stats = []
    for brand in INDIAN_BRANDS:
        mask    = df_scored["brand"].str.contains(brand, na=False)
        subset  = df_scored[mask]
        if len(subset) < 3:
            continue
        brand_stats.append({
            "brand":           brand.title(),
            "products":        len(subset),
            "avg_health":      round(subset["health"].mean()),
            "avg_deception":   round(subset["deception"].mean()),
            "worst_product":   subset.loc[subset["health"].idxmin(), "product"],
            "most_deceptive":  subset.loc[subset["deception"].idxmax(), "product"],
            "grade_dist":      subset["grade"].value_counts().to_dict(),
        })

    brand_stats.sort(key=lambda x: x["avg_deception"], reverse=True)

    # ── TOP DECEPTIVE PRODUCTS ────────────────────────────
    top_deceptive = (
        df_scored[df_scored["deception"] > 40]
        .sort_values("deception", ascending=False)
        .head(20)
        [["product","brand","health","grade","deception","claims"]]
        .to_dict("records")
    )

    # ── CATEGORY ANALYSIS ─────────────────────────────────
    if "categories" in df.columns:
        df_scored["categories"] = df["categories"].fillna("")
        cat_health = (
            df_scored.groupby("categories")["health"]
            .agg(["mean","count"])
            .query("count >= 10")
            .sort_values("mean")
            .head(10)
            .reset_index()
            .rename(columns={"mean":"avg_health","count":"products"})
        )
        worst_categories = cat_health.to_dict("records")
    else:
        worst_categories = []

    # ── OVERALL STATS ─────────────────────────────────────
    overall = {
        "total_products":      len(df_scored),
        "avg_health_score":    round(df_scored["health"].mean()),
        "pct_grade_d_or_e":    round((df_scored["grade"].isin(["D","E"])).mean() * 100),
        "pct_deceptive":       round((df_scored["d_level"] == "deceptive").mean() * 100),
        "pct_misleading":      round((df_scored["d_level"] == "misleading").mean() * 100),
    }

    output = {
        "overall":          overall,
        "brand_stats":      brand_stats[:15],
        "top_deceptive":    top_deceptive,
        "worst_categories": worst_categories,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Brand analysis saved → {OUT_FILE}")
    print(f"\nKey findings:")
    print(f"  • {overall['pct_grade_d_or_e']}% of Indian packaged foods score D or E")
    print(f"  • {overall['pct_deceptive']}% are actively deceptive")
    print(f"  • {overall['pct_misleading']}% are misleading")
    print(f"\nMost deceptive brands:")
    for b in brand_stats[:5]:
        print(f"  • {b['brand']}: avg deception {b['avg_deception']}/100")


if __name__ == "__main__":
    run_analysis()