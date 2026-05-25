# ============================================================
#  api/main.py  —  FoodLens Backend API
#  Run: uvicorn api.main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from database import get_cached, save_cache, log_scan
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ifhi import compute_ifhi, ifhi_to_grade, compute_percent_rda
from scorer import (
    engineer_features,
    compute_confidence,
    compute_deception_score,
    check_red_flag_ingredients,
    detect_serving_manipulation,
    generate_plain_explanation,
    build_nutrient_breakdown,
    grade_to_display,
    ALL_FEATURE_NAMES,
)
from lookup import lookup_barcode, find_alternatives

# ── LOAD MODEL ────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "food_scorer.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model files not found. Run Colab training first.")
    model  = None
    scaler = None

app = FastAPI(title="FoodLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── REQUEST MODELS ────────────────────────────────────────
class ManualEntry(BaseModel):
    product_name:       Optional[str]   = "Unknown product"
    ingredients:        Optional[str]   = ""
    categories:         Optional[str]   = ""
    energy_100g:        Optional[float] = 0
    sugars_100g:        Optional[float] = 0
    saturated_fat_100g: Optional[float] = 0
    fat_100g:           Optional[float] = 0
    sodium_100g:        Optional[float] = 0
    fiber_100g:         Optional[float] = 0
    proteins_100g:      Optional[float] = 0
    additives_n:        Optional[int]   = 0
    nova_group:         Optional[int]   = 0
    serving_size_g:     Optional[float] = None
    package_size_g:     Optional[float] = None


# ── CORE SCORING ──────────────────────────────────────────
def run_scoring(nutrients, product_name, ingredients,
                categories, serving_size_g=None,
                package_size_g=None, image_url="",
                brand="", alternatives=None):

    if not model or not scaler:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")

    # ML prediction
    feat    = engineer_features(nutrients)
    feat_sc = scaler.transform(feat)
    ml_raw  = float(model.predict(feat_sc)[0])
    ml_raw  = max(0, min(100, ml_raw))

    # Formula score
    formula_score = compute_ifhi(nutrients)
    formula_grade = ifhi_to_grade(formula_score)

    # ML score — clamp to valid range then grade
    ml_score = int(round(ml_raw))
    ml_grade = ifhi_to_grade(ml_score)

    # Use formula score as primary (ML validates it)
    final_score = formula_score
    final_grade = formula_grade

    confidence  = compute_confidence(nutrients)
    deception   = compute_deception_score(
        product_name, ingredients, final_score, categories
    )
    red_flags   = check_red_flag_ingredients(ingredients)
    explanation = generate_plain_explanation(nutrients, final_score)
    breakdown   = build_nutrient_breakdown(nutrients)
    serving_m   = detect_serving_manipulation(
        serving_size_g, package_size_g, nutrients
    )
    pct_rda     = compute_percent_rda(nutrients)

    return {
        "product": {
            "name":      product_name,
            "brand":     brand,
            "image_url": image_url,
        },
        "score": {
            "ifhi":        final_score,
            "grade":       final_grade,
            "grade_label": grade_to_display(final_grade)["label"],
            "grade_color": grade_to_display(final_grade)["color"],
            "ml_raw":      round(ml_raw, 1),
            "ml_grade":    ml_grade,
        },
        "deception":    deception,
        "confidence":   confidence,
        "breakdown":    breakdown,
        "explanation":  explanation,
        "red_flags":    red_flags,
        "pct_rda":      pct_rda,
        "serving":      serving_m if serving_m.get("detected") else None,
        "alternatives": alternatives or [],
    }


# ── ROUTES ────────────────────────────────────────────────
@app.get("/")
def root():
    landing = os.path.join(FRONTEND_DIR, "landing.html")
    if os.path.exists(landing):
        return FileResponse(landing)
    return {"message": "FoodLens API running"}
@app.get("/app")
def app_page():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "App not found"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/scan/{barcode}")
def scan(barcode: str):
    # Check cache first
    cached = get_cached(barcode)
    if cached:
        return cached

    # Not cached — hit Open Food Facts
    try:
        product = lookup_barcode(barcode)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not product:
        raise HTTPException(
            status_code=404,
            detail="Product not found in Open Food Facts database"
        )

    nutrients = product["nutrients"]
    try:
        alts = find_alternatives(
            product.get("category_tag", ""),
            compute_ifhi(nutrients)
        )
    except Exception:
        alts = []

    result = run_scoring(
        nutrients    = nutrients,
        product_name = product["product_name"],
        ingredients  = product.get("ingredients_text", ""),
        categories   = product.get("categories", ""),
        image_url    = product.get("image_url", ""),
        brand        = product.get("brand", ""),
        alternatives = alts,
    )

    # Save to cache and log the scan
    save_cache(barcode, product["product_name"], result)
    log_scan(barcode, product["product_name"],
             result["score"]["ifhi"], result["deception"]["score"])

    return result


@app.post("/score")
def score_manual(data: ManualEntry):
    nutrients = {
        "energy_100g":        data.energy_100g,
        "sugars_100g":        data.sugars_100g,
        "saturated-fat_100g": data.saturated_fat_100g,
        "fat_100g":           data.fat_100g,
        "sodium_100g":        data.sodium_100g,
        "fiber_100g":         data.fiber_100g,
        "proteins_100g":      data.proteins_100g,
        "additives_n":        data.additives_n,
        "nova_group":         data.nova_group,
    }
    return run_scoring(
        nutrients      = nutrients,
        product_name   = data.product_name,
        ingredients    = data.ingredients,
        categories     = data.categories,
        serving_size_g = data.serving_size_g,
        package_size_g = data.package_size_g,
    )