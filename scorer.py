# ============================================================
#  scorer.py  —  Feature engineering, IFHI scoring,
#                deception detection, confidence,
#                plain-English explanations
#
#  Shared by: train_model.py, colab_notebook.py, api/main.py
# ============================================================

import numpy as np
from ifhi import (
    compute_ifhi, ifhi_to_grade, compute_percent_rda,
    FSSAI_HIGH_THRESHOLDS, ICMR_NIN_RDA,
)

# ── FEATURE ORDER ─────────────────────────────────────────
# This exact order must match during training AND inference.
# Any change here requires full model retrain.
FEATURE_COLS = [
    "energy_100g",
    "sugars_100g",
    "saturated-fat_100g",
    "fat_100g",
    "sodium_100g",
    "fiber_100g",
    "proteins_100g",
    "additives_n",
    "nova_group",
]

NUTRIENT_DISPLAY_NAMES = {
    "energy_100g":        "Energy",
    "sugars_100g":        "Sugar",
    "saturated-fat_100g": "Saturated fat",
    "fat_100g":           "Total fat",
    "sodium_100g":        "Sodium",
    "fiber_100g":         "Dietary fibre",
    "proteins_100g":      "Protein",
    "additives_n":        "Additives",
    "nova_group":         "Processing (NOVA)",
}

# ── HEALTH CLAIM KEYWORDS ─────────────────────────────────
# Used by compute_deception_score()
# Weight = how strongly the word implies healthiness
HEALTH_CLAIM_WORDS = {
    # Weight 2 — strong, specific health claims
    "immunity":          2, "protein rich":    2, "high protein":    2,
    "superfood":         2, "zero sugar":       2, "no added sugar":  2,
    "sugar free":        2, "fat free":         2, "low fat":         2,
    "diabetic":          2, "heart healthy":    2, "clinically":      2,
    "scientifically":    2, "doctors recommend":2, "pediatrician":    2,
    "kids":              2, "children":         2, "school":          2,
    "tiffin":            2, "mummy":            2, "mom":             2,
    "loved by kids":     2,

    # Weight 1 — moderate, general health implications
    "natural":           1, "organic":          1, "multigrain":      1,
    "wholegrain":        1, "whole grain":      1, "oats":            1,
    "fortified":         1, "enriched":         1, "vitamins":        1,
    "minerals":          1, "calcium":          1, "iron":            1,
    "fiber":             1, "fibre":            1, "probiotic":       1,
    "prebiotic":         1, "antioxidant":      1, "omega":           1,
    "wellness":          1, "goodness":         1, "wholesome":       1,
    "nutritious":        1, "nourishing":       1, "healthy":         1,
    "light":             1, "lite":             1, "diet":            1,
    "fitness":           1, "energy":           1, "growth":          1,
    "smart":             1, "active":           1, "fresh":           1,
    "tasty treat":       1, "family":           1, "2 minute":        1,
    "nutrition":         1, "fruit":            1,
}

HIGH_RISK_CATEGORIES = [
    "instant noodles", "chips", "biscuits", "cookies",
    "namkeen", "snacks", "wafers", "chocolate drink",
    "health drink", "energy drink", "fruit juice",
]

# ── INGREDIENT RED FLAGS ──────────────────────────────────
RED_FLAG_INGREDIENTS = {
    "partially hydrogenated": {
        "severity": "high",
        "reason":   "Contains trans fats — linked to heart disease, banned in many countries"
    },
    "vanaspati": {
        "severity": "high",
        "reason":   "Hydrogenated vegetable fat — primary source of trans fats in Indian foods"
    },
    "high fructose corn syrup": {
        "severity": "high",
        "reason":   "Industrial sweetener strongly linked to obesity and fatty liver disease"
    },
    "tbhq": {
        "severity": "high",
        "reason":   "Synthetic antioxidant — restricted in Japan and other countries"
    },
    "sodium benzoate": {
        "severity": "medium",
        "reason":   "Can form benzene (carcinogen) when combined with Vitamin C"
    },
    "e102": {
        "severity": "medium",
        "reason":   "Tartrazine — artificial colour linked to hyperactivity in children"
    },
    "e110": {
        "severity": "medium",
        "reason":   "Sunset Yellow — banned in Norway and Finland"
    },
    "e951": {
        "severity": "medium",
        "reason":   "Aspartame — artificial sweetener, controversial long-term effects"
    },
    "carrageenan": {
        "severity": "medium",
        "reason":   "Linked to gut inflammation in some peer-reviewed studies"
    },
    "bleached flour": {
        "severity": "low",
        "reason":   "Chemically whitened, stripped of natural nutrients and fibre"
    },
    "refined palm oil": {
        "severity": "low",
        "reason":   "High in saturated fat; significant environmental impact"
    },
    "maida": {
        "severity": "low",
        "reason":   "Refined wheat flour — very high glycaemic index, negligible fibre"
    },
}


# ── FEATURE ENGINEERING ───────────────────────────────────
def engineer_features(nutrients: dict) -> np.ndarray:
    """
    Build the complete feature vector for the ML model.

    9 raw features + 4 engineered interaction features = 13 total.

    The 4 engineered features capture nutrient interactions that
    a simple threshold system cannot — they are the primary reason
    the model outperforms a rule-based lookup.

    Returns np.ndarray of shape (1, 13).
    """
    row = [float(nutrients.get(col, 0) or 0) for col in FEATURE_COLS]
    energy, sugars, sat_fat, fat, sodium, fiber, protein, additives, nova = row

    # Sugar is metabolically worse when fibre is low —
    # fibre slows glucose absorption (ICMR-NIN dietary advice [2])
    sugar_to_fiber = sugars / (fiber + 0.1)

    # What proportion of total fat is the unhealthy saturated kind
    # (avocado = high fat, low saturation = OK;
    #  vanaspati = high fat, high saturation = bad)
    fat_saturation = sat_fat / (fat + 0.1)

    # How much protein relative to calories —
    # high energy + low protein = nutrient-poor
    protein_energy = protein / (energy + 1)

    # Binary flag for NOVA Group 4 ultra-processed foods
    # Given stronger signal as separate feature because of
    # India-specific NCD burden from ultra-processing [5][6]
    is_ultra = 1.0 if nova == 4 else 0.0

    all_features = row + [sugar_to_fiber, fat_saturation,
                          protein_energy, is_ultra]
    return np.array(all_features, dtype=np.float32).reshape(1, -1)


ALL_FEATURE_NAMES = FEATURE_COLS + [
    "sugar_to_fiber_ratio",
    "fat_saturation_ratio",
    "protein_energy_ratio",
    "is_ultra_processed",
]


# ── GRADE DISPLAY ─────────────────────────────────────────
def grade_to_display(grade: str) -> dict:
    return {
        "A": {"color": "#1a7f37", "label": "Excellent", "emoji": "🟢"},
        "B": {"color": "#5ca000", "label": "Good",      "emoji": "🟡"},
        "C": {"color": "#efb700", "label": "Moderate",  "emoji": "🟠"},
        "D": {"color": "#e86b00", "label": "Poor",      "emoji": "🔴"},
        "E": {"color": "#cc1515", "label": "Very Poor", "emoji": "🔴"},
    }.get(grade, {"color": "#888", "label": "Unknown",  "emoji": "⚪"})


# ── DECEPTION SCORE ───────────────────────────────────────
def compute_deception_score(product_name: str,
                             ingredients: str,
                             ifhi_score: int,
                             categories: str = "") -> dict:
    """
    Measures the gap between how healthy a product CLAIMS to be
    versus how healthy it ACTUALLY is (per IFHI).

    Returns dict with score (0-100), level, color, emoji,
    claims found, and human-readable explanation.
    """
    text         = (product_name + " " + ingredients).lower()
    claims       = []
    total_weight = 0

    for keyword, weight in HEALTH_CLAIM_WORDS.items():
        if keyword in text:
            claims.append(keyword)
            total_weight += weight

    # Category implicit deception:
    # products in high-risk categories marketed to families/children
    # carry inherent deception through positioning even without
    # explicit health words
    cat_lower         = categories.lower()
    in_risky_category = any(c in cat_lower for c in HIGH_RISK_CATEGORIES)
    if in_risky_category and ifhi_score < 50:
        total_weight += 3

    max_possible   = sum(HEALTH_CLAIM_WORDS.values())
    claim_strength = min(total_weight / (max_possible * 0.3), 1.0)
    actual_badness = (100 - ifhi_score) / 100

    # Deception = strong claims × poor actual score
    # A product with no claims and poor score = bad food, not deceptive
    # A product with strong claims and poor score = actively deceptive
    score = min(round(claim_strength * actual_badness * 100), 100)

    if score >= 60:
        level = "deceptive";  color = "#cc1515"; emoji = "🚨"
    elif score >= 30:
        level = "misleading"; color = "#e86b00"; emoji = "⚠️"
    else:
        level = "honest";     color = "#1a7f37"; emoji = "✅"

    if claims and actual_badness > 0.4:
        explanation = (
            f"This product uses {len(claims)} health/family claim(s) "
            f"({', '.join(claims[:3])}{'...' if len(claims) > 3 else ''}) "
            f"but scores only {ifhi_score}/100 on actual nutrition. "
            f"The marketing significantly overstates its health value."
        )
    elif in_risky_category and ifhi_score < 50 and not claims:
        explanation = (
            f"No explicit health claims detected, but this product belongs "
            f"to a high-risk category often marketed to families while "
            f"scoring poorly on nutrition ({ifhi_score}/100)."
        )
    elif not claims:
        explanation = "No health claims detected. Score reflects actual nutrition only."
    else:
        explanation = (
            f"Health claims found ({', '.join(claims[:3])}) are roughly "
            f"consistent with the actual nutrition score ({ifhi_score}/100)."
        )

    return {
        "score":             score,
        "level":             level,
        "color":             color,
        "emoji":             emoji,
        "claims":            claims,
        "explanation":       explanation,
        "in_risky_category": in_risky_category,
    }


# ── INGREDIENT RED FLAGS ──────────────────────────────────
def check_red_flag_ingredients(ingredients: str) -> list:
    text  = ingredients.lower()
    flags = []
    for ingredient, info in RED_FLAG_INGREDIENTS.items():
        if ingredient in text:
            flags.append({
                "ingredient": ingredient,
                "severity":   info["severity"],
                "reason":     info["reason"],
            })
    order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda x: order[x["severity"]])
    return flags


# ── SERVING SIZE MANIPULATION ─────────────────────────────
def detect_serving_manipulation(serving_size_g,
                                 package_size_g,
                                 nutrients) -> dict:
    if not serving_size_g or not package_size_g:
        return {"detected": False}
    servings = package_size_g / serving_size_g
    if servings <= 2.5:
        return {"detected": False}

    sodium_mg     = nutrients.get("sodium_100g", 0) * (package_size_g/100) * 1000
    sugar_g       = nutrients.get("sugars_100g",  0) * (package_size_g/100)
    calories_kcal = nutrients.get("energy_100g",  0) * (package_size_g/100)

    return {
        "detected":     True,
        "servings":     round(servings, 1),
        "serving_size": serving_size_g,
        "package_size": package_size_g,
        "message": (
            f"Serving size is only {serving_size_g}g but packet is "
            f"{package_size_g}g ({round(servings, 1)} servings). "
            f"Eating the whole packet = {round(calories_kcal)} kcal, "
            f"{round(sugar_g, 1)}g sugar, {round(sodium_mg)}mg sodium."
        ),
    }


# ── PLAIN ENGLISH EXPLANATION ─────────────────────────────
def generate_plain_explanation(nutrients: dict,
                                ifhi_score: int) -> list:
    """
    Convert nutrient values into plain English sentences using
    ICMR-NIN % Daily Values [2].
    Skips values that are clearly unrealistic (OFF data errors).
    """
    lines  = []
    pct_rv = compute_percent_rda(nutrients)

    SANITY_LIMITS = {
        "sodium_100g":        3.0,    # g/100g — above this = data error
        "sugars_100g":        100.0,
        "saturated-fat_100g": 100.0,
        "fiber_100g":         100.0,
        "proteins_100g":      100.0,
    }

    for nut_key, info in pct_rv.items():
        val  = info["value"]
        pct  = info["pct_rda"]
        name = info["name"]

        if val == 0:
            continue

        # Skip unrealistic values — flag as data quality issue
        limit = SANITY_LIMITS.get(nut_key)
        if limit and val > limit:
            lines.append(
                f"⚠️ **{name}**: reported value ({val:.2f}g/100g) appears "
                f"unrealistic — possible data error in database. "
                f"Check the nutrition label directly."
            )
            continue

        # Negative nutrients — high is bad
        if nut_key in ("sugars_100g", "sodium_100g", "saturated-fat_100g"):
            if pct > 50:
                lines.append(
                    f"🔴 **{name}**: {val:.1f}{'mg' if nut_key=='sodium_100g' else 'g'}"
                    f" per 100g — {round(pct)}% of ICMR-NIN daily limit "
                    f"in just 100g"
                )
            elif pct > 20:
                lines.append(
                    f"⚠️ **{name}**: {val:.1f}g per 100g — "
                    f"moderately high ({round(pct)}% of daily limit)"
                )

        # Positive nutrients — high is good
        elif nut_key in ("fiber_100g", "proteins_100g"):
            if pct > 15:
                lines.append(
                    f"✅ **{name}**: {val:.1f}g per 100g — "
                    f"good source ({round(pct)}% of ICMR-NIN daily needs)"
                )

    nova = int(nutrients.get("nova_group", 0) or 0)
    if nova == 4:
        lines.append(
            "🔴 **Ultra-processed (NOVA Group 4)** — regardless of "
            "individual nutrient values, ultra-processed foods are "
            "independently linked to diabetes, obesity, and cardiovascular "
            "disease in peer-reviewed studies [Monteiro et al., BMJ 2019]"
        )
    elif nova == 3:
        lines.append(
            "⚠️ **Processed food (NOVA Group 3)** — "
            "contains additives and preservatives"
        )
    elif 1 <= nova <= 2:
        lines.append(
            "✅ **Minimally processed (NOVA Group 1–2)** — "
            "close to its natural form"
        )

    additives = int(nutrients.get("additives_n", 0) or 0)
    if additives >= 5:
        lines.append(
            f"⚠️ **{additives} additives** detected — "
            f"high additive count is a marker of ultra-processing"
        )

    return lines


# ── NUTRIENT BREAKDOWN TABLE ──────────────────────────────
def build_nutrient_breakdown(nutrients: dict) -> list:
    """
    Returns per-nutrient assessment using FSSAI FOPL
    thresholds [1] for display in the result card.
    """
    rows = []

    breakdown_config = {
        "sugars_100g": {
            "name":      "Sugar",
            "low":       5.0,
            "high":      FSSAI_HIGH_THRESHOLDS["sugars_100g"],
            "direction": "lower_is_better",
            "unit":      "g",
        },
        "saturated-fat_100g": {
            "name":      "Saturated fat",
            "low":       1.5,
            "high":      FSSAI_HIGH_THRESHOLDS["saturated_fat_100g"],
            "direction": "lower_is_better",
            "unit":      "g",
        },
        "sodium_100g": {
            "name":      "Sodium",
            "low":       0.12,
            "high":      FSSAI_HIGH_THRESHOLDS["sodium_100g"],
            "direction": "lower_is_better",
            "unit":      "g",
            "note":      "FSSAI HIGH: >600mg/100g [1]",
        },
        "fiber_100g": {
            "name":      "Dietary fibre",
            "low":       1.5,
            "high":      3.0,
            "direction": "higher_is_better",
            "unit":      "g",
            "note":      "ICMR-NIN RDA: 40g/day [2]",
        },
        "proteins_100g": {
            "name":      "Protein",
            "low":       2.0,
            "high":      5.0,
            "direction": "higher_is_better",
            "unit":      "g",
            "note":      "ICMR-NIN RDA: 50g/day [2]",
        },
    }

    for col, cfg in breakdown_config.items():
        v  = float(nutrients.get(col, 0) or 0)
        lo = cfg["low"]
        hi = cfg["high"]
        d  = cfg["direction"]

        if d == "lower_is_better":
            level = "low" if v < lo else ("medium" if v < hi else "high")
        else:
            level = "high" if v >= hi else ("medium" if v >= lo else "low")

        rows.append({
            "name":      cfg["name"],
            "value":     f"{v:.1f}{cfg['unit']}",
            "level":     level,
            "direction": d,
            "note":      cfg.get("note", ""),
        })

    nova = int(nutrients.get("nova_group", 0) or 0)
    if nova:
        nova_labels = {
            1: "Unprocessed",
            2: "Culinary ingredient",
            3: "Processed",
            4: "Ultra-processed",
        }
        nova_levels = {1: "high", 2: "high", 3: "medium", 4: "low"}
        rows.append({
            "name":      "Processing (NOVA)",
            "value":     f"Group {nova} · {nova_labels.get(nova, '')}",
            "level":     nova_levels.get(nova, "medium"),
            "direction": "higher_is_better",
            "note":      "NOVA classification [5]",
        })

    return rows


# ── CONFIDENCE ────────────────────────────────────────────
TRAINING_RANGES = {
    "energy_100g":        (0, 700),
    "sugars_100g":        (0, 50),
    "saturated-fat_100g": (0, 25),
    "fat_100g":           (0, 60),
    "sodium_100g":        (0, 3),
    "fiber_100g":         (0, 20),
    "proteins_100g":      (0, 40),
}


def compute_confidence(nutrients: dict) -> dict:
    """
    Assess score reliability based on:
    1. How many of 9 core nutrient fields are present
    2. Whether values fall within realistic training ranges
    3. Explicit check for known OFF data errors (sodium > 3g)
    """
    # Hard check for unrealistic sodium (known OFF data error pattern)
    sodium = float(nutrients.get("sodium_100g", 0) or 0)
    if sodium > 3.0:
        return {
            "level":   "low",
            "filled":  0,
            "total":   9,
            "message": (
                f"⚠️ Suspicious data — sodium value ({sodium:.2f}g/100g) "
                f"is physically unrealistic. Score may be inaccurate. "
                f"Check the nutrition label directly."
            ),
        }

    filled = sum(
        1 for col in FEATURE_COLS
        if nutrients.get(col) not in (None, 0, "", 0.0)
    )
    out_of_range = sum(
        1 for col, (lo, hi) in TRAINING_RANGES.items()
        if nutrients.get(col) and
        not (lo <= float(nutrients[col]) <= hi)
    )

    data_score  = filled / len(FEATURE_COLS)
    range_score = 1 - (out_of_range / len(TRAINING_RANGES))
    combined    = (data_score * 0.6) + (range_score * 0.4)

    if combined >= 0.75:
        return {
            "level":   "high",
            "filled":  filled,
            "total":   9,
            "message": f"High confidence — {filled}/9 nutrient fields complete",
        }
    elif combined >= 0.5:
        return {
            "level":   "medium",
            "filled":  filled,
            "total":   9,
            "message": f"Moderate confidence — {filled}/9 fields available",
        }
    else:
        return {
            "level":   "low",
            "filled":  filled,
            "total":   9,
            "message": f"Low confidence — only {filled}/9 fields found",
        }