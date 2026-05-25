# ============================================================
#  ocr.py  —  Extract nutrient values from a label photo
#  Requires: pip install pytesseract pillow
#  Mac:      brew install tesseract
#  Linux:    sudo apt install tesseract-ocr
# ============================================================

import re
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Regex patterns to find "Nutrient Name ... number unit" in OCR text
PATTERNS = {
    "energy_100g":        r"energy[^\d]*(\d+\.?\d*)\s*kcal",
    "sugars_100g":        r"sugar[s]?[^\d]*(\d+\.?\d*)\s*g",
    "saturated-fat_100g": r"saturated\s*fat[^\d]*(\d+\.?\d*)\s*g",
    "fat_100g":           r"(?:total\s*)?fat[^\d]*(\d+\.?\d*)\s*g",
    "sodium_100g":        r"sodium[^\d]*(\d+\.?\d*)\s*(?:g|mg)",
    "fiber_100g":         r"(?:dietary\s*)?fi[b]?re?[^\d]*(\d+\.?\d*)\s*g",
    "proteins_100g":      r"protein[s]?[^\d]*(\d+\.?\d*)\s*g",
}

SODIUM_MG_PATTERN = r"sodium[^\d]*(\d+\.?\d*)\s*mg"


def extract_nutrients_from_image(image_file) -> dict:
    """
    Takes a PIL Image or file-like object, runs OCR, and
    extracts nutrient values per 100g using regex patterns.

    Returns a dict of nutrient values (partial — only what's found).
    """
    if not OCR_AVAILABLE:
        raise ImportError(
            "pytesseract not installed. Run: pip install pytesseract pillow\n"
            "and on Mac: brew install tesseract"
        )

    img  = Image.open(image_file) if not isinstance(image_file, Image.Image) else image_file
    text = pytesseract.image_to_string(img).lower()

    nutrients = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            # Convert mg → g for sodium
            if key == "sodium_100g" and re.search(SODIUM_MG_PATTERN, text, re.IGNORECASE):
                val = val / 1000
            nutrients[key] = val

    return nutrients


def ocr_confidence(nutrients: dict) -> int:
    """Returns the % of key nutrients successfully extracted (0-100)."""
    found = sum(1 for k in PATTERNS if k in nutrients)
    return int(round(found / len(PATTERNS) * 100))
