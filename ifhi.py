# ============================================================
#  ifhi.py  —  India Food Health Index (IFHI)
#
#  NOT an officially recognised standard.
#  An ML training target and consumer scoring function
#  derived entirely from official Indian regulatory and
#  nutritional science sources listed below.
#
#  CITATIONS
#  ─────────────────────────────────────────────────────────
#  [1] FSSAI Draft Food Safety and Standards (Labelling and
#      Display) Amendment Regulations, 2022.
#      Front of Pack Nutrition Labelling (FOPL) thresholds.
#      fssai.gov.in
#
#  [2] ICMR-NIN. "Nutrient Requirements for Indians —
#      Recommended Dietary Allowances (RDA) and Estimated
#      Average Requirements (EAR)." 2020.
#      National Institute of Nutrition, Hyderabad.
#      nin.res.in
#
#  [3] GBD 2017 Diet Collaborators. "Health effects of
#      dietary risks in 195 countries, 1990–2017."
#      The Lancet, 393(10184), 2019.
#      doi:10.1016/S0140-6736(19)30041-8
#
#  [4] India State-Level Disease Burden Initiative.
#      "Nations within a nation: variations in epidemiological
#      transition across the states of India, 1990–2016."
#      The Lancet, 390(10111), 2017.
#      doi:10.1016/S0140-6736(17)32804-0
#
#  [5] Monteiro CA et al. "Ultra-processed foods: what they
#      are and how to identify them."
#      Public Health Nutrition, 22(5), 2019.
#      doi:10.1017/S1368980018003762
#
#  [6] Srour B et al. "Ultra-processed food intake and risk
#      of cardiovascular disease: prospective cohort study."
#      BMJ, 365, 2019. doi:10.1136/bmj.l1451
#
#  REFERENCE INDIVIDUAL
#  ─────────────────────────────────────────────────────────
#  60kg Indian adult, sedentary-to-moderate activity level.
#  Per ICMR-NIN 2020 [2], Table 1.
#
#  DESIGN NOTE on protein cap
#  ─────────────────────────────────────────────────────────
#  Protein bonus is capped when sugar > 22.5g/100g.
#  Justification: India has 77M T2 diabetics [3]. A product
#  with 37g sugar per 100g causes acute glycaemic harm
#  regardless of protein content. Adding protein to a
#  high-sugar product is a common marketing strategy
#  (e.g. health drink powders) that should not offset
#  the primary negative factor. This is consistent with
#  FSSAI's FOPL philosophy [1] where HIGH sugar triggers
#  a warning label that overrides any positive claims.
# ============================================================

ICMR_NIN_RDA = {
    "energy_kcal":     2000,
    "protein_g":       50,
    "fat_g":           55,
    "saturated_fat_g": 20,
    "carbohydrate_g":  300,
    "sugars_g":        25,
    "sodium_mg":       2000,
    "dietary_fibre_g": 40,
    "calcium_mg":      1000,
    "iron_mg":         17,
}

FSSAI_HIGH_THRESHOLDS = {
    "sugars_100g":        12.5,
    "sodium_100g":        0.6,
    "saturated_fat_100g": 5.0,
    "total_fat_100g":     20.0,
}

WEIGHT_RATIONALE = {
    "sodium":       "Hypertension #1 diet-related NCD in India [4]",
    "sugar":        "T2 diabetes affects 77M Indians; sugar main driver [3]",
    "saturated_fat":"Cardiovascular disease 2nd leading cause of death [4]",
    "nova":         "Ultra-processing independently linked to NCDs [5][6]",
    "energy":       "Obesity rising; energy density proxy for poor quality",
    "fibre":        "Average Indian consumes 15-20g vs 40g recommended [3]",
    "protein":      "Protein deficiency affects majority of Indian adults [2]",
}


def compute_ifhi(nutrients: dict) -> int:
    """
    Compute India Food Health Index (IFHI) for a product.

    Parameters
    ----------
    nutrients : dict with keys matching FEATURE_COLS in scorer.py
                Values are per 100g as printed on nutrition label.

    Returns
    -------
    int : IFHI score 0-100 (100 = optimal for Indian adult health)

    Scoring logic
    -------------
    Start at 100. Deduct for nutrients of concern per FSSAI [1]
    and ICMR-NIN [2] thresholds. Add back for beneficial nutrients.
    Protein bonus is capped when sugar is critically high — see
    design note above. Final score clamped to [0, 100].
    """
    score = 100

    # ── SODIUM ───────────────────────────────────────────────
    # ICMR-NIN RDA: 2000mg/day [2]
    # FSSAI HIGH flag: >600mg/100g [1]
    # Strong penalty: 28% of Indian adults have hypertension [4]
    sodium_mg = float(nutrients.get("sodium_100g", 0) or 0) * 1000
    if sodium_mg > 1200:    score -= 30
    elif sodium_mg > 600:   score -= 20
    elif sodium_mg > 300:   score -= 10
    elif sodium_mg > 120:   score -= 4

    # ── SUGAR ────────────────────────────────────────────────
    # ICMR-NIN free sugars: <25g/day [2]
    # FSSAI HIGH flag: >12.5g/100g [1]
    # 77M Indians have T2 diabetes — sugar is primary dietary driver [3]
    # Penalty is stronger at extreme levels (>22.5g = >90% daily limit)
    sugar = float(nutrients.get("sugars_100g", 0) or 0)
    if sugar > 30.0:        score -= 35   # extreme — >120% daily limit
    elif sugar > 22.5:      score -= 28   # very high — >90% daily limit
    elif sugar > 12.5:      score -= 16   # FSSAI HIGH threshold [1]
    elif sugar > 7.5:       score -= 8
    elif sugar > 5.0:       score -= 4

    # ── SATURATED FAT ────────────────────────────────────────
    # ICMR-NIN: <20g/day [2], FSSAI HIGH: >5g/100g [1]
    # CVD is 2nd leading cause of death in India [4]
    sat_fat = float(nutrients.get("saturated-fat_100g", 0) or 0)
    if sat_fat > 8.0:       score -= 20
    elif sat_fat > 5.0:     score -= 14
    elif sat_fat > 2.5:     score -= 7
    elif sat_fat > 1.5:     score -= 3

    # ── TOTAL FAT ────────────────────────────────────────────
    # ICMR-NIN: 55g/day [2], FSSAI HIGH: >20g/100g [1]
    total_fat = float(nutrients.get("fat_100g", 0) or 0)
    if total_fat > 25.0:    score -= 10
    elif total_fat > 20.0:  score -= 6
    elif total_fat > 12.0:  score -= 2

    # ── ENERGY DENSITY ───────────────────────────────────────
    # Reference: 2000 kcal/day for 60kg Indian adult [2]
    energy = float(nutrients.get("energy_100g", 0) or 0)
    if energy > 500:        score -= 10
    elif energy > 400:      score -= 6
    elif energy > 300:      score -= 2

    # ── NOVA ULTRA-PROCESSING ────────────────────────────────
    # NOVA classification [5], Indian NCD evidence [6]
    # Stronger penalty than Nutri-Score — Indian market grew 10x
    # in 20 years, population more metabolically vulnerable
    nova = int(nutrients.get("nova_group", 0) or 0)
    if nova == 4:           score -= 20
    elif nova == 3:         score -= 7

    # ── ADDITIVES ────────────────────────────────────────────
    # Proxy for ultra-processing when NOVA is unavailable
    additives = int(nutrients.get("additives_n", 0) or 0)
    if additives >= 10:     score -= 10
    elif additives >= 5:    score -= 5
    elif additives >= 2:    score -= 2

    # ── PROTEIN (positive) ───────────────────────────────────
    # ICMR-NIN RDA: 50g/day [2]
    # Indian diets are chronically protein-deficient [2]
    #
    # CAP: protein bonus is reduced to max +5 when sugar > 22.5g/100g
    # Rationale: high-sugar products with added protein (health drink
    # powders like Horlicks/Bournvita) use protein as a marketing claim
    # to offset sugar concerns. FSSAI FOPL [1] treats HIGH sugar as a
    # warning that overrides positive claims. A product cannot be
    # considered beneficial for protein when it delivers dangerous
    # sugar levels — the net metabolic effect is negative.
    protein = float(nutrients.get("proteins_100g", 0) or 0)
    if sugar > 22.5:
        # Protein bonus capped at +5 when sugar is critically high
        if protein >= 5:    score += 5
        elif protein >= 2:  score += 2
    else:
        if protein >= 12:   score += 15
        elif protein >= 8:  score += 10
        elif protein >= 5:  score += 5
        elif protein >= 2:  score += 2

    # ── DIETARY FIBRE (positive) ─────────────────────────────
    # ICMR-NIN RDA: 40g/day [2]
    # Avg Indian consumes only 15-20g — less than half needed [3]
    fibre = float(nutrients.get("fiber_100g", 0) or 0)
    if fibre >= 8:          score += 15
    elif fibre >= 5:        score += 10
    elif fibre >= 3:        score += 5
    elif fibre >= 1.5:      score += 2

    return max(0, min(100, round(score)))


def ifhi_to_grade(score: int) -> str:
    """
    Convert IFHI score to letter grade.
    Thresholds reflect FSSAI traffic light philosophy [1].
    """
    if score >= 75: return "A"
    if score >= 60: return "B"
    if score >= 45: return "C"
    if score >= 30: return "D"
    return "E"


def compute_percent_rda(nutrients: dict) -> dict:
    """
    Compute % of ICMR-NIN daily reference values [2]
    for each nutrient per 100g. Used for plain-English
    explanations in the app.
    """
    mapping = {
        "energy_100g":        ("energy_kcal",     "Energy"),
        "proteins_100g":      ("protein_g",       "Protein"),
        "sugars_100g":        ("sugars_g",        "Sugar"),
        "saturated-fat_100g": ("saturated_fat_g", "Saturated fat"),
        "sodium_100g":        ("sodium_mg",       "Sodium"),
        "fiber_100g":         ("dietary_fibre_g", "Dietary fibre"),
    }
    result = {}
    for nut_key, (rda_key, display_name) in mapping.items():
        val = float(nutrients.get(nut_key, 0) or 0)
        if rda_key == "sodium_mg":
            val = val * 1000
        rda = ICMR_NIN_RDA[rda_key]
        result[nut_key] = {
            "name":    display_name,
            "value":   val,
            "rda":     rda,
            "pct_rda": round(val / rda * 100, 1) if rda > 0 else 0,
        }
    return result


def validate_ifhi_with_known_products():
    """
    Sanity check: run IFHI on products with known nutritional
    profiles. All must pass before training proceeds.
    """
    test_cases = [
        {
            "name": "Brown rice cooked — expected A/B (>= 70)",
            "nutrients": {
                "energy_100g":130, "sugars_100g":0.1,
                "saturated-fat_100g":0.1, "fat_100g":0.3,
                "sodium_100g":0.001, "fiber_100g":1.8,
                "proteins_100g":2.7, "additives_n":0, "nova_group":1
            },
            "expected_min": 70,
        },
        {
            "name": "Plain oats — expected A (>= 75)",
            "nutrients": {
                "energy_100g":380, "sugars_100g":1.0,
                "saturated-fat_100g":1.4, "fat_100g":7.0,
                "sodium_100g":0.004, "fiber_100g":10.0,
                "proteins_100g":13.0, "additives_n":0, "nova_group":1
            },
            "expected_min": 75,
        },
        {
            "name": "Instant noodles — expected C/D (<= 55)",
            "nutrients": {
                "energy_100g":385, "sugars_100g":2.1,
                "saturated-fat_100g":6.5, "fat_100g":14.0,
                "sodium_100g":0.82, "fiber_100g":1.2,
                "proteins_100g":9.0, "additives_n":5, "nova_group":4
            },
            "expected_max": 55,
        },
        {
            "name": "Health drink Horlicks-like — expected D/E (<= 45)",
            "nutrients": {
                "energy_100g":380, "sugars_100g":37.0,
                "saturated-fat_100g":1.2, "fat_100g":3.5,
                "sodium_100g":0.28, "fiber_100g":0.5,
                "proteins_100g":14.0, "additives_n":8, "nova_group":4
            },
            "expected_max": 45,
        },
    ]

    print("IFHI Validation — Sanity Checks")
    print("=" * 55)
    all_passed = True
    for tc in test_cases:
        score  = compute_ifhi(tc["nutrients"])
        grade  = ifhi_to_grade(score)
        passed = True
        if "expected_min" in tc and score < tc["expected_min"]:
            passed = False
        if "expected_max" in tc and score > tc["expected_max"]:
            passed = False
        if not passed:
            all_passed = False
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {tc['name']}")
        print(f"         Score: {score}/100  Grade: {grade}")
    print("=" * 55)
    if all_passed:
        print("All sanity checks passed. Safe to proceed with training.")
    else:
        print("Some checks FAILED — review formula weights.")
    return all_passed


if __name__ == "__main__":
    validate_ifhi_with_known_products()