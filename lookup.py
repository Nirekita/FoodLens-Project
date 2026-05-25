# ============================================================
#  lookup.py  —  Open Food Facts API barcode lookup
# ============================================================

import requests

OFF_API    = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
HEADERS    = {"User-Agent": "NutriLens/2.0 (college-project; india)"}

NUTRIENT_MAP = {
    "energy-kcal_100g":   "energy_100g",
    "energy_100g":        "energy_100g",
    "sugars_100g":        "sugars_100g",
    "saturated-fat_100g": "saturated-fat_100g",
    "fat_100g":           "fat_100g",
    "sodium_100g":        "sodium_100g",
    "fiber_100g":         "fiber_100g",
    "proteins_100g":      "proteins_100g",
}


def lookup_barcode(barcode: str) -> dict | None:
    try:
        resp = requests.get(
            OFF_API.format(barcode=barcode.strip()),
            timeout=8, headers=HEADERS,
        )
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"API request failed: {e}")

    if data.get("status") != 1:
        return None

    p          = data["product"]
    nutriments = p.get("nutriments", {})
    nutrients  = {}

    for off_key, our_key in NUTRIENT_MAP.items():
        val = nutriments.get(off_key)
        if val is not None:
            try:
                nutrients[our_key] = float(val)
            except (ValueError, TypeError):
                pass

    try:
        nutrients["nova_group"] = int(p.get("nova_group"))
    except (TypeError, ValueError):
        pass

    additives_tags           = p.get("additives_tags", [])
    nutrients["additives_n"] = len(additives_tags)

    return {
        "product_name":     p.get("product_name") or
                            p.get("product_name_en", "Unknown"),
        "brand":            p.get("brands", "Unknown"),
        "image_url":        p.get("image_front_url") or
                            p.get("image_url", ""),
        "ingredients_text": p.get("ingredients_text", ""),
        "nutrients":        nutrients,
        "nova_group":       nutrients.get("nova_group"),
        "additives_n":      len(additives_tags),
        "additives_tags":   [t.replace("en:", "") for t in additives_tags],
        "categories":       p.get("categories", ""),
        "category_tag":     (p.get("categories_tags") or [""])[0],
        "serving_size":     p.get("serving_size", ""),
        "quantity":         p.get("quantity", ""),
    }


def find_alternatives(category_tag: str,
                       current_ifhi: int,
                       n: int = 3) -> list:
    """
    Search OFF for products in the same category with better
    nutrition. Uses nutriscore_score as proxy since IFHI is
    our own index (not stored in OFF).
    Better nutriscore → likely better IFHI too.
    """
    if not category_tag:
        return []
    try:
        resp = requests.get(
            OFF_SEARCH, timeout=8, headers=HEADERS,
            params={
                "action":         "process",
                "tagtype_0":      "categories",
                "tag_contains_0": "contains",
                "tag_0":          category_tag,
                "sort_by":        "nutriscore_score",
                "page_size":      15,
                "json":           1,
                "countries":      "india",
            },
        )
        items = resp.json().get("products", [])
    except Exception:
        return []

    from ifhi import compute_ifhi
    from scorer import NUTRIENT_MAP

    alts = []
    for p in items:
        nutriments = p.get("nutriments", {})
        nutrients  = {}
        for off_key, our_key in NUTRIENT_MAP.items():
            val = nutriments.get(off_key)
            if val is not None:
                try:
                    nutrients[our_key] = float(val)
                except (ValueError, TypeError):
                    pass
        try:
            nutrients["nova_group"] = int(p.get("nova_group"))
        except (TypeError, ValueError):
            pass
        nutrients["additives_n"] = len(p.get("additives_tags", []))

        alt_ifhi = compute_ifhi(nutrients)
        if alt_ifhi > current_ifhi + 5:
            from ifhi import ifhi_to_grade
            alts.append({
                "name":  p.get("product_name", "Unknown")[:60],
                "brand": p.get("brands", "")[:30],
                "grade": ifhi_to_grade(alt_ifhi),
                "ifhi":  alt_ifhi,
            })
        if len(alts) >= n:
            break

    alts.sort(key=lambda x: x["ifhi"], reverse=True)
    return alts