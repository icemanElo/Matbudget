"""
utils.py - Version 3.0 med SQLite
Fördelar över CSV:
- Snabbare queries (index)
- ACID-transaktioner
- Bättre för stora datamängder
- Enklare uppdateringar/borttagningar
"""

UTILS_VERSION = "v3-sqlite"

import os
import json
import re
import sqlite3
import pandas as pd
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager

# =============================================================================
# PATHS
# =============================================================================
DB_PATH = "matbudget.db"
RULES_PATH = "category_rules.json"

# =============================================================================
# PRE-COMPILED REGEX
# =============================================================================
_RE_UNITS = re.compile(r"\b\d+[,.]?\d*\s*(G|GR|KG|HG|ML|CL|DL|L|ST|STYCK|PCS|PACK|PKT)\b", re.IGNORECASE)
_RE_PERCENT = re.compile(r"\b\d+[,.]?\d*\s*%\b")
_RE_SPECIAL_CHARS = re.compile(r"[^A-ZÅÄÖ0-9 ]")
_RE_WHITESPACE = re.compile(r"\s+")

_NOISE_WORDS = frozenset({
    "EKO", "EKOLOGISK", "KRAV", "SVENSK", "SVERIGE", "IMPORT",
    "KLASS", "CLASS", "I", "1", "SE", "ES", "NL", "PL"
})

_STOPWORDS_KEYWORDS = frozenset({
    "EKO", "EKOLOGISK", "KRAV", "SVENSK", "SVERIGE", "IMPORT", "MAX", "MAX2",
    "KG", "G", "GR", "L", "DL", "CL", "ML", "ST", "PKT", "PACK",
    "PRIS", "KR", "SEK", "PLUS"
})

DEFAULT_CATEGORIES = [
    "Livsmedel/Grönt", "Livsmedel/Frukt", "Livsmedel/Mejeri", "Livsmedel/Kött",
    "Livsmedel/Fisk", "Livsmedel/Vegetariskt", "Livsmedel/Bröd", "Livsmedel/Fryst",
    "Livsmedel/Skafferi", "Livsmedel/Snacks", "Dryck", "Hygien", "Hushåll",
    "Djur", "Pant", "Rabatt", "Övrigt", "Okategoriserat",
]

# =============================================================================
# DATABASE SCHEMA
# =============================================================================
_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS artiklar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datum DATE,
    butik TEXT,
    vara TEXT,
    pris REAL DEFAULT 0,
    kategori TEXT DEFAULT 'Okategoriserat',
    ean TEXT,
    rad INTEGER,
    kvitto_id TEXT,
    fil TEXT,
    antal INTEGER DEFAULT 1,
    mangd REAL DEFAULT 0,
    enhet TEXT DEFAULT 'st',
    kategori_last INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_datum ON artiklar(datum);
CREATE INDEX IF NOT EXISTS idx_butik ON artiklar(butik);
CREATE INDEX IF NOT EXISTS idx_kategori ON artiklar(kategori);
CREATE INDEX IF NOT EXISTS idx_kvitto_id ON artiklar(kvitto_id);
CREATE INDEX IF NOT EXISTS idx_vara ON artiklar(vara);
"""


@contextmanager
def get_db_connection():
    """Context manager för databasanslutning."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initiera databasen med schema."""
    with get_db_connection() as conn:
        conn.executescript(_DB_SCHEMA)


def migrate_from_csv(csv_path: str = "matutgifter.csv"):
    """Migrera data från gammal CSV till SQLite."""
    if not os.path.exists(csv_path):
        return False
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return False
        
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        column_map = {
            "datum": "datum", "butik": "butik", "vara": "vara", "pris": "pris",
            "kategori": "kategori", "ean": "ean", "rad": "rad", "kvittoid": "kvitto_id",
            "fil": "fil", "antal": "antal", "mangd": "mangd", "enhet": "enhet",
            "kategori_låst": "kategori_last"
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        with get_db_connection() as conn:
            df.to_sql("artiklar", conn, if_exists="append", index=False)
        
        os.rename(csv_path, csv_path + ".backup")
        return True
        
    except Exception as e:
        print(f"Migration error: {e}")
        return False


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
def load_db() -> pd.DataFrame:
    """Ladda all data som DataFrame."""
    init_db()
    
    if os.path.exists("matutgifter.csv") and not _table_has_data():
        migrate_from_csv()
    
    with get_db_connection() as conn:
        df = pd.read_sql_query("""
            SELECT 
                id, datum as Datum, butik as Butik, vara as Vara, pris as Pris,
                kategori as Kategori, ean as EAN, rad as Rad, kvitto_id as KvittoID,
                fil as Fil, antal as Antal, mangd as Mangd, enhet as Enhet,
                kategori_last as Kategori_Låst
            FROM artiklar ORDER BY datum DESC
        """, conn)
    
    if not df.empty:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df["Pris"] = pd.to_numeric(df["Pris"], errors="coerce").fillna(0.0)
        df["Antal"] = pd.to_numeric(df["Antal"], errors="coerce").fillna(1).astype(int)
    
    return df


def _table_has_data() -> bool:
    try:
        with get_db_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM artiklar").fetchone()
            return result[0] > 0
    except:
        return False


def save_db(df: pd.DataFrame) -> None:
    """Spara DataFrame till databasen (ersätter all data)."""
    init_db()
    
    df_save = df.copy()
    column_map = {
        "Datum": "datum", "Butik": "butik", "Vara": "vara", "Pris": "pris",
        "Kategori": "kategori", "EAN": "ean", "Rad": "rad", "KvittoID": "kvitto_id",
        "Fil": "fil", "Antal": "antal", "Mangd": "mangd", "Enhet": "enhet",
        "Kategori_Låst": "kategori_last"
    }
    
    df_save = df_save.rename(columns={k: v for k, v in column_map.items() if k in df_save.columns})
    
    valid_cols = ["datum", "butik", "vara", "pris", "kategori", "ean", "rad", 
                  "kvitto_id", "fil", "antal", "mangd", "enhet", "kategori_last"]
    df_save = df_save[[c for c in df_save.columns if c in valid_cols]]
    
    with get_db_connection() as conn:
        conn.execute("DELETE FROM artiklar")
        df_save.to_sql("artiklar", conn, if_exists="append", index=False)


def insert_rows(rows: List[Dict]) -> int:
    """Lägg till nya rader (effektivare än save_db för nya data)."""
    if not rows:
        return 0
    
    init_db()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        inserted = 0
        
        for row in rows:
            existing = cursor.execute("""
                SELECT id FROM artiklar WHERE kvitto_id = ? AND rad = ? AND pris = ?
            """, (row.get("KvittoID"), row.get("Rad"), row.get("Pris"))).fetchone()
            
            if existing:
                continue
            
            cursor.execute("""
                INSERT INTO artiklar (datum, butik, vara, pris, kategori, ean, rad, kvitto_id, fil, antal, mangd, enhet)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("Datum"), row.get("Butik"), row.get("Vara"), row.get("Pris", 0),
                row.get("Kategori", "Okategoriserat"), row.get("EAN"), row.get("Rad"),
                row.get("KvittoID"), row.get("Fil"), row.get("Antal", 1),
                row.get("Mangd", 0), row.get("Enhet", "st")
            ))
            inserted += 1
        
        return inserted


def update_kategori(vara_norm: str, new_kategori: str) -> int:
    """Uppdatera kategori för alla rader med given VaraNorm."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT id, vara FROM artiklar")
        ids_to_update = [row["id"] for row in cursor if normalize_item_name(row["vara"]) == vara_norm]
        
        if ids_to_update:
            placeholders = ",".join("?" * len(ids_to_update))
            conn.execute(f"UPDATE artiklar SET kategori = ? WHERE id IN ({placeholders})",
                        [new_kategori] + ids_to_update)
        
        return len(ids_to_update)


def delete_by_butik(butik: str) -> int:
    """Ta bort alla rader för en butik."""
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM artiklar WHERE butik = ?", (butik,))
        return cursor.rowcount


def delete_all() -> None:
    """Ta bort all data."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM artiklar")


def get_stats() -> Dict[str, Any]:
    """Hämta snabbstatistik direkt från databasen."""
    with get_db_connection() as conn:
        return {
            "total_rows": conn.execute("SELECT COUNT(*) FROM artiklar").fetchone()[0],
            "total_spend": conn.execute(
                "SELECT COALESCE(SUM(pris), 0) FROM artiklar WHERE pris > 0 AND kategori NOT IN ('Pant', 'Rabatt')"
            ).fetchone()[0],
            "total_discount": abs(conn.execute(
                "SELECT COALESCE(SUM(pris), 0) FROM artiklar WHERE pris < 0"
            ).fetchone()[0]),
            "uncategorized": conn.execute(
                "SELECT COUNT(*) FROM artiklar WHERE kategori = 'Okategoriserat'"
            ).fetchone()[0],
            "unique_products": conn.execute(
                "SELECT COUNT(DISTINCT vara) FROM artiklar"
            ).fetchone()[0],
        }


# =============================================================================
# RULES (JSON - oförändrat)
# =============================================================================
_rules_cache: Dict[str, Any] = {}
_rules_mtime: float = 0.0


def _invalidate_rules_cache() -> None:
    global _rules_cache, _rules_mtime
    _rules_cache = {}
    _rules_mtime = 0.0
    normalize_name.cache_clear()
    canonical_item_name.cache_clear()


def _default_rules() -> Dict[str, Any]:
    return {
        "categories": DEFAULT_CATEGORIES.copy(),
        "ean_rules": {},
        "name_rules": {},
        "keyword_rules": [],
        "normalization": {"exact": {}, "contains": []},
        "special_rules": {
            "pant_keywords": ["PANT"],
            "discount_keywords": ["RABATT", "PRISAVDRAG", "WILLYS PLUS", "NEDSATT", "PRISSÄNKT"],
        },
    }


def load_rules() -> Dict[str, Any]:
    global _rules_cache, _rules_mtime

    if os.path.exists(RULES_PATH):
        current_mtime = os.path.getmtime(RULES_PATH)
        if _rules_cache and current_mtime == _rules_mtime:
            return _rules_cache
        _rules_mtime = current_mtime

    if not os.path.exists(RULES_PATH):
        rules = _default_rules()
        save_rules(rules)
        _rules_cache = _ensure_special_defaults(rules)
        return _rules_cache

    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        rules = _default_rules()

    repaired = _repair_rules(rules)
    meta = rules.setdefault("_meta", {})
    
    if not meta.get("name_rules_canonical_v1"):
        nr = rules.get("name_rules", {}) or {}
        migrated = {normalize_name(canonical_item_name(k)): v for k, v in nr.items() if normalize_name(canonical_item_name(k))}
        rules["name_rules"] = migrated
        meta["name_rules_canonical_v1"] = True
        repaired = True

    rules = _ensure_special_defaults(rules)
    if repaired:
        save_rules(rules)

    _rules_cache = rules
    return rules


def _repair_rules(rules: Dict[str, Any]) -> bool:
    repaired = False
    base = _default_rules()

    for k, v in base.items():
        if k not in rules:
            rules[k] = v
            repaired = True

    for key, expected_type in [("keyword_rules", list), ("ean_rules", dict), ("name_rules", dict)]:
        if not isinstance(rules.get(key), expected_type):
            rules[key] = [] if expected_type == list else {}
            repaired = True

    if not isinstance(rules.get("normalization"), dict):
        rules["normalization"] = {"exact": {}, "contains": []}
        repaired = True

    return repaired


def _ensure_special_defaults(rules: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(rules, dict):
        rules = {}

    sr = rules.setdefault("special_rules", {})
    if not isinstance(sr, dict):
        sr = {}
        rules["special_rules"] = sr

    if not isinstance(sr.get("pant_keywords"), list) or not sr.get("pant_keywords"):
        sr["pant_keywords"] = ["PANT"]

    if not isinstance(sr.get("discount_keywords"), list) or not sr.get("discount_keywords"):
        sr["discount_keywords"] = ["RABATT", "PRISAVDRAG", "AVDRAG", "KAMPANJ", "ERBJUDANDE", "BONUS", "PLUS", "MEDLEM", "MEDLEMSPRIS"]

    return rules


def save_rules(rules: Dict[str, Any]) -> None:
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    _invalidate_rules_cache()


# =============================================================================
# NORMALIZATION
# =============================================================================
@lru_cache(maxsize=4096)
def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = _RE_SPECIAL_CHARS.sub(" ", str(name).upper())
    return _RE_WHITESPACE.sub(" ", n).strip()


@lru_cache(maxsize=4096)
def canonical_item_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).upper().strip()
    s = _RE_UNITS.sub(" ", s)
    s = _RE_PERCENT.sub(" ", s)
    s = _RE_SPECIAL_CHARS.sub(" ", s)
    tokens = sorted([t for t in s.split() if t not in _NOISE_WORDS and len(t) > 1])
    result = " ".join(tokens).strip()
    return result if result else str(name).upper()


def normalize_item_name(name: str, rules: Optional[Dict[str, Any]] = None) -> str:
    if name is None:
        return ""
    if rules is None:
        rules = load_rules()

    raw_up = str(name).upper().strip()
    canon_up = canonical_item_name(raw_up)
    norm_cfg = rules.get("normalization", {}) or {}
    exact = norm_cfg.get("exact", {}) or {}

    if raw_up in exact:
        return str(exact[raw_up]).upper().strip()
    if canon_up in exact:
        return str(exact[canon_up]).upper().strip()

    for r in norm_cfg.get("contains", []) or []:
        kw = str(r.get("keyword", "")).upper().strip()
        to = str(r.get("to", "")).upper().strip()
        if kw and to and (kw in raw_up or kw in canon_up):
            return to

    return canon_up


# =============================================================================
# CATEGORIZATION
# =============================================================================
def is_pant_line(name: str, rules: Optional[Dict[str, Any]] = None) -> bool:
    rules = _ensure_special_defaults(rules or load_rules())
    up = normalize_name(canonical_item_name(str(name)))
    return any(normalize_name(k) in up for k in rules.get("special_rules", {}).get("pant_keywords", []) if k)


def is_discount_line(name: str, price: Optional[float] = None, rules: Optional[Dict[str, Any]] = None) -> bool:
    rules = _ensure_special_defaults(rules or load_rules())
    up = normalize_name(canonical_item_name(str(name)))

    if price is not None:
        try:
            if float(price) < 0 and not is_pant_line(name, rules):
                return True
        except (ValueError, TypeError):
            pass

    if "PANT" in up:
        return False

    return any(normalize_name(k) in up for k in rules.get("special_rules", {}).get("discount_keywords", []) if k)


def is_special_line(name: str, rules: dict) -> bool:
    return is_pant_line(name, rules) or is_discount_line(name, rules=rules)


def categorize_item(name: str, ean: Optional[str] = None, price: Optional[float] = None,
                    rules: Optional[Dict[str, Any]] = None) -> str:
    rules = rules or load_rules()
    up = normalize_name(canonical_item_name(name))

    if is_pant_line(name, rules):
        return "Pant"
    if is_discount_line(name, price, rules):
        return "Rabatt"

    if ean:
        e = str(ean).strip()
        if e in rules.get("ean_rules", {}):
            return rules["ean_rules"][e]

    if up and up in rules.get("name_rules", {}):
        return rules["name_rules"][up]

    for r in rules.get("keyword_rules", []):
        kw = normalize_name(r.get("keyword", ""))
        cat = r.get("category", "")
        if kw and cat and kw in up:
            return cat

    return "Okategoriserat"


def trace_categorization(name: str, ean: Optional[str] = None, price: Optional[float] = None,
                         rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rules = rules or load_rules()
    canon = canonical_item_name(name)
    up = normalize_name(canon)
    steps = []

    def _step(stage: str, matched: bool, detail: Any = None):
        steps.append({"stage": stage, "matched": matched, "detail": detail})

    pant_hit = next((normalize_name(k) for k in rules.get("special_rules", {}).get("pant_keywords", []) if normalize_name(k) in up), None)
    _step("special:pant", pant_hit is not None, pant_hit)

    disc_hit = next((normalize_name(k) for k in rules.get("special_rules", {}).get("discount_keywords", []) if normalize_name(k) in up), None)
    _step("special:discount", disc_hit is not None, disc_hit)

    ean_rule = rules.get("ean_rules", {}).get(str(ean).strip()) if ean else None
    _step("rule:ean", ean_rule is not None, ean_rule)

    name_rule = rules.get("name_rules", {}).get(up)
    _step("rule:name", name_rule is not None, name_rule)

    keyword_matches = [
        {"idx": idx, "keyword": normalize_name(r.get("keyword", "")), "category": r.get("category", "")}
        for idx, r in enumerate(rules.get("keyword_rules", []))
        if normalize_name(r.get("keyword", "")) in up and r.get("category")
    ]
    _step("rule:keyword", len(keyword_matches) > 0, f"{len(keyword_matches)} match(es)")

    kw_counts = {}
    for r in rules.get("keyword_rules", []):
        kw = normalize_name(r.get("keyword", ""))
        if kw:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1

    return {
        "final": categorize_item(name, ean, price, rules),
        "canon": canon, "up": up, "steps": steps,
        "keyword_matches": keyword_matches,
        "blocking": {"ean_rule": ean_rule, "name_rule": name_rule},
        "duplicates": {"keyword_duplicates": {k: c for k, c in kw_counts.items() if c > 1}},
    }


# =============================================================================
# DATAFRAME OPERATIONS
# =============================================================================
def recategorize_dataframe(df: pd.DataFrame, force: bool = False, respect_locked: bool = True) -> pd.DataFrame:
    rules = load_rules()

    for col in ["Kategori", "EAN", "Vara", "Pris"]:
        if col not in df.columns:
            df[col] = ""

    if "Kategori_Låst" not in df.columns:
        df["Kategori_Låst"] = False

    def _cat(row):
        curr = str(row.get("Kategori", ""))
        locked = bool(row.get("Kategori_Låst", False))
        if respect_locked and locked and curr:
            return curr
        if not force and curr and curr != "Okategoriserat":
            return curr
        return categorize_item(str(row.get("Vara", "")), ean=row.get("EAN"), price=row.get("Pris"), rules=rules)

    df["Kategori"] = df.apply(_cat, axis=1)
    return df


# =============================================================================
# RULE CRUD
# =============================================================================
def add_ean_rule(ean: str, category: str) -> None:
    r = load_rules()
    if ean:
        r["ean_rules"][str(ean).strip()] = category
    save_rules(r)


def add_name_rule(name: str, category: str) -> None:
    r = load_rules()
    n = normalize_name(canonical_item_name(name))
    if n:
        r["name_rules"][n] = category
    save_rules(r)


def add_keyword_rule(keyword: str, category: str) -> None:
    r = load_rules()
    kw = normalize_name(keyword)
    if not kw:
        return

    rules_list = list(r.get("keyword_rules", []))
    same_kw_idx = [i for i, rule in enumerate(rules_list) if normalize_name(rule.get("keyword", "")) == kw]

    if same_kw_idx:
        new_list = [rule for i, rule in enumerate(rules_list) if i not in same_kw_idx]
        new_list.insert(0, {"keyword": kw, "category": category})
        r["keyword_rules"] = new_list
    else:
        rules_list.insert(0, {"keyword": kw, "category": category})
        r["keyword_rules"] = rules_list

    save_rules(r)


def delete_ean_rule(ean: str) -> None:
    r = load_rules()
    r["ean_rules"].pop(str(ean).strip(), None)
    save_rules(r)


def delete_name_rule(name: str) -> None:
    r = load_rules()
    r["name_rules"].pop(normalize_name(canonical_item_name(name)), None)
    save_rules(r)


def delete_keyword_rule(idx: int) -> None:
    r = load_rules()
    if 0 <= idx < len(r["keyword_rules"]):
        r["keyword_rules"].pop(idx)
        save_rules(r)


def update_keyword_rule(idx: int, keyword: str, category: str) -> None:
    r = load_rules()
    if 0 <= idx < len(r["keyword_rules"]):
        r["keyword_rules"][idx] = {"keyword": normalize_name(keyword), "category": category}
        save_rules(r)


def cleanup_rules(rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = rules or load_rules()
    seen = set()
    new_list = []
    for rule in r.get("keyword_rules", []):
        kw = normalize_name(rule.get("keyword", ""))
        cat = rule.get("category", "")
        if kw and cat and kw not in seen:
            seen.add(kw)
            new_list.append({"keyword": kw, "category": cat})
    r["keyword_rules"] = new_list
    r.setdefault("name_rules", {})
    r.setdefault("ean_rules", {})
    r.setdefault("special_rules", {"pant_keywords": [], "discount_keywords": []})
    r.setdefault("categories", [])
    save_rules(r)
    return r


# =============================================================================
# SUGGESTIONS
# =============================================================================
def suggest_category(name: str, ean: Optional[str], df: Optional[pd.DataFrame],
                     rules: Optional[Dict[str, Any]] = None) -> Tuple[str, str, float]:
    rules = rules or load_rules()

    reg_cat = categorize_item(name=name, ean=ean, price=None, rules=rules)
    if reg_cat != "Okategoriserat":
        return reg_cat, "Regelträff", 1.0

    if df is None or df.empty or "Kategori" not in df.columns or "Vara" not in df.columns:
        return "Okategoriserat", "Ingen match", 0.0

    if "VaraNorm" in df.columns:
        vnorm = normalize_item_name(name)
        m = df[(df["VaraNorm"].astype(str).str.upper() == str(vnorm).upper()) & (df["Kategori"] != "Okategoriserat")]
        if not m.empty:
            return m["Kategori"].value_counts().idxmax(), "Historik (VaraNorm)", 0.75

    canon = canonical_item_name(name)
    tokens = [t for t in normalize_name(canon).split() if len(t) >= 4][:3]

    if tokens:
        vnorm_series = df["Vara"].astype(str).apply(lambda s: normalize_name(canonical_item_name(s)))
        mask = vnorm_series.str.contains("|".join(re.escape(t) for t in tokens), regex=True)
        m = df[mask & (df["Kategori"] != "Okategoriserat")]
        if not m.empty:
            return m["Kategori"].value_counts().idxmax(), f"Historik (token)", 0.7

    return "Okategoriserat", "Ingen match", 0.0


# =============================================================================
# RULE HEALTH
# =============================================================================
def rule_health_report(rules: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    r = rules or load_rules()
    rep = {"counts": {}, "issues": {}, "examples": {}}

    rep["counts"] = {
        "ean_rules": len(r.get("ean_rules", {}) or {}),
        "name_rules": len(r.get("name_rules", {}) or {}),
        "keyword_rules": len(r.get("keyword_rules", []) or []),
        "categories": len(r.get("categories", []) or []),
        "norm_exact": len((r.get("normalization", {}) or {}).get("exact", {}) or {}),
    }

    kw_counts = {}
    for rr in r.get("keyword_rules", []) or []:
        kw = normalize_name(rr.get("keyword", ""))
        if kw:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1

    rep["issues"]["keyword_duplicates"] = {k: c for k, c in kw_counts.items() if c > 1}
    rep["issues"]["name_rules_to_okategoriserat"] = [k for k, v in (r.get("name_rules", {}) or {}).items() if v == "Okategoriserat"]
    rep["issues"]["ean_rules_to_okategoriserat"] = [k for k, v in (r.get("ean_rules", {}) or {}).items() if v == "Okategoriserat"]

    return rep


def apply_rule_health_fixes(fix_keyword_duplicates: bool = True, remove_okategoriserat_exact_rules: bool = True,
                            rules: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    r = rules or load_rules()

    if remove_okategoriserat_exact_rules:
        r["name_rules"] = {k: v for k, v in (r.get("name_rules", {}) or {}).items() if v != "Okategoriserat"}
        r["ean_rules"] = {k: v for k, v in (r.get("ean_rules", {}) or {}).items() if v != "Okategoriserat"}

    if fix_keyword_duplicates:
        r = cleanup_rules(r)

    save_rules(r)
    return r
