"""
parsers.py - Optimerad version
Förbättringar:
- Kompilerade regex-mönster (en gång, inte vid varje anrop)
- Förenklad och mer läsbar kod
- Bättre felhantering
- Reducerad kodduplicering
"""

import re
import hashlib
import pdfplumber
from io import BytesIO
from typing import List, Dict, Any, Optional, Callable

# =============================================================================
# PRE-COMPILED REGEX PATTERNS
# =============================================================================

# Gemensamma patterns
_RE_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_RE_NUMBER = re.compile(r"-?\d+\.\d+|-?\d+")

# Multiplier/quantity patterns
_RE_MULT_WITH_UNIT = re.compile(r"\b(\d+)\s*(?:st|styck|pcs)\s*[*xX]\s*[\d,\.]+", re.IGNORECASE)
_RE_MULT_NO_SPACE = re.compile(r"\b(\d+)(?:st|styck|pcs)\s*[*xX]\s*[\d,\.]+", re.IGNORECASE)

# Willys cleanup patterns
_RE_WILLYS_MULT1 = re.compile(r"\b\d+\s*(?:ST|STYCK|PCS)\s*[*xX]\s*[\d,\.]+", re.IGNORECASE)
_RE_WILLYS_MULT2 = re.compile(r"\b\d+(?:ST|STYCK|PCS)\s*[*xX]\s*[\d,\.]+", re.IGNORECASE)
_RE_WILLYS_MULT3 = re.compile(r"\b\d+\s*[*xX]\s*[\d,\.]+")
_RE_WILLYS_TAIL = re.compile(r"[*xX]\s*[\d,\.]+\s*$")
_RE_WHITESPACE = re.compile(r"\s+")

# ICA patterns
_RE_ICA_START = re.compile(r"Beskrivning\s+Artikelnummer\s+Pris\s+Mängd\s+Summa", re.IGNORECASE)
_RE_ICA_END = re.compile(r"Betalat|Moms %|Betalningsinformation", re.IGNORECASE)
_RE_ICA_PANT = re.compile(r"^\*?\s*Pant.*?(-?\d+[,\.]\d{2})\s*$", re.IGNORECASE)
_RE_ICA_ST = re.compile(r"^(.+?)\s+(\d{8,14})\s+([\d,.-]+)\s+([\d,.-]+)\s+st\s+([\d,.-]+)$", re.IGNORECASE)
_RE_ICA_KG = re.compile(r"^(.+?)\s+(\d{8,14})\s+([\d,.-]+)\s+([\d,.-]+)\s+kg\s+([\d,.-]+)$", re.IGNORECASE)
_RE_ICA_RABATT = re.compile(r"^(.+?)\s+(-[\d,.]+)$", re.IGNORECASE)
_RE_ICA_PANT_ANYWHERE = re.compile(r"^\*?\s*Pant.*?(-?\d+[,\.]\d{2})\s*$", re.IGNORECASE | re.MULTILINE)

# COOP patterns
_RE_COOP_START = re.compile(r"Org\.?\s*Nr", re.IGNORECASE)
_RE_COOP_END = re.compile(r"^Total\s+SEK\s+[\d,.]+", re.IGNORECASE | re.MULTILINE)
_RE_COOP_PRICE = re.compile(r"^(.+?)\s+(-?\d+[,\.]\d{2})$", re.IGNORECASE)
_RE_COOP_MULT = re.compile(r"^\d+([,\.]\d+)?\s*(STK|ST|KG)\s*x\s*\d+[,\.]\d{2}$", re.IGNORECASE)

# Willys patterns
_RE_WILLYS_START = re.compile(r"=+\s*Start\s+Självscanning\s*=+", re.IGNORECASE)
_RE_WILLYS_END = re.compile(r"=+\s*Slut\s+Självscanning\s*=+", re.IGNORECASE)
_RE_WILLYS_PRICE = re.compile(r"^(.+?)\s+(-?\d+[,\.]\d{2})$")
_RE_WILLYS_WEIGHT = re.compile(r"^(\d+[,\.]\d+)\s*kg\*\s*(\d+[,\.]\d+)\s*kr/kg\s+(-?\d+[,\.]\d{2})$", re.IGNORECASE)
_RE_WILLYS_DISCOUNT = re.compile(r"^(Willys Plus:.*|Rabatt:.*)\s+(-?\d+[,\.]\d{2})$", re.IGNORECASE)
_RE_HAS_LETTERS = re.compile(r"[A-ZÅÄÖa-zåäö]")
_RE_STARTS_DIGIT = re.compile(r"^\d")

# Store detection
_RE_WILLYS = re.compile(r"\bWILLYS\b", re.IGNORECASE)
_RE_COOP = re.compile(r"\bCOOP\b", re.IGNORECASE)
_RE_ICA = re.compile(r"\bICA\b", re.IGNORECASE)

# Ignore patterns (as frozensets for O(1) lookup)
_COOP_IGNORE = frozenset([
    "ANTAL ARTIKLAR", "ERHÅLLNA", "ERHALLNA", "POÄNG", "POANG",
    "MEDLEMSKORT", "MOMS", "NETTO", "BRUTTO", "BETALA", "TOTAL", "KORT"
])

_WILLYS_IGNORE = frozenset([
    "TOTAL", "VAROR", "SEK", "SPARA KVITTOT", "ÖPPETTIDER", "VÄLKOMMEN",
    "KASSA", "MOMS", "NETTO", "BRUTTO", "KÖP", "REF:", "TERM:", "TVR:", "AID:",
    "MASTERCARD", "DEBIT", "KONTOKORT", "MOTTAGET"
])


# =============================================================================
# HELPERS
# =============================================================================

def parse_num(val: Any) -> float:
    """Parse numbers with comma/dot. Returns 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        s = str(val).replace(" ", "").replace(",", ".")
        matches = _RE_NUMBER.findall(s)
        return float(matches[-1]) if matches else 0.0
    except Exception:
        return 0.0


def warn(msg: str, debug: bool = False, on_warn: Optional[Callable] = None) -> None:
    """Log warning to console and optionally to UI."""
    try:
        print(f"[WARN] {msg}")
    except Exception:
        pass

    if debug and on_warn:
        try:
            on_warn(msg)
        except Exception:
            pass


def extract_multiplier_qty(raw_name: str) -> int:
    """Extract quantity multiplier from product name (e.g., '3st*74,90' -> 3)."""
    if not raw_name:
        return 1

    s = str(raw_name)

    # Try patterns in order of specificity
    for pattern in [_RE_MULT_WITH_UNIT, _RE_MULT_NO_SPACE]:
        m = pattern.search(s)
        if m:
            return int(m.group(1))

    return 1


def clean_willys_name(name: str) -> str:
    """Clean Willys product name by removing multipliers and trailing prices."""
    if not name:
        return ""

    n = _RE_WHITESPACE.sub(" ", name.strip())

    # Remove multiplier patterns
    for pattern in [_RE_WILLYS_MULT1, _RE_WILLYS_MULT2, _RE_WILLYS_MULT3]:
        n = pattern.sub("", n)

    # Remove trailing price patterns
    n = _RE_WILLYS_TAIL.sub("", n)

    return _RE_WHITESPACE.sub(" ", n).strip(" -•\t")


def _contains_any(text: str, keywords: frozenset) -> bool:
    """Check if text contains any of the keywords."""
    return any(kw in text for kw in keywords)


def _make_row(date: str, store: str, row_num: int, name: str, price: float,
              category: str, antal: int = 1, **kwargs) -> Dict[str, Any]:
    """Create a standardized row dict."""
    row = {
        "Datum": date,
        "Butik": store,
        "Rad": row_num,
        "Vara": name,
        "Antal": antal,
        "Pris": price,
        "Kategori": category,
    }
    row.update(kwargs)
    return row


# =============================================================================
# ICA PARSER
# =============================================================================

def parse_ica_text(pdf_name: str, date: str, text: str, categorize_func: Callable,
                   debug: bool = False, on_warn: Optional[Callable] = None) -> List[Dict]:
    """Parse ICA receipt text."""
    rows = []

    start = _RE_ICA_START.search(text)
    end = _RE_ICA_END.search(text)

    if not start or not end or end.start() <= start.end():
        warn(f"{pdf_name}: ICA: kunde inte hitta start/end för varulista", debug, on_warn)
        return rows

    block = text[start.end():end.start()]
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    for i, ln in enumerate(lines, start=1):
        # Pant line
        m = _RE_ICA_PANT.match(ln)
        if m:
            pant_sum = parse_num(m.group(1))
            if abs(pant_sum) > 1e-9:
                rows.append(_make_row(date, "ICA", i, "Pant", pant_sum, "Pant"))
            continue

        # Item with st (pieces)
        m = _RE_ICA_ST.match(ln)
        if m:
            name = m.group(1).strip().lstrip("*")
            summa = parse_num(m.group(5))
            if abs(summa) > 1e-9:
                rows.append(_make_row(
                    date, "ICA", i, name, summa, categorize_func(name),
                    antal=int(round(parse_num(m.group(4)) or 1)),
                    EAN=m.group(2), Mangd=0.0, Enhet="st"
                ))
            continue

        # Item with kg (weight)
        m = _RE_ICA_KG.match(ln)
        if m:
            name = m.group(1).strip().lstrip("*")
            summa = parse_num(m.group(5))
            if abs(summa) > 1e-9:
                rows.append(_make_row(
                    date, "ICA", i, name, summa, categorize_func(name),
                    EAN=m.group(2), Mangd=parse_num(m.group(4)), Enhet="kg"
                ))
            continue

        # Discount line
        m = _RE_ICA_RABATT.match(ln)
        if m:
            price = parse_num(m.group(2))
            if price < 0:
                rows.append(_make_row(date, "ICA", i, m.group(1).strip(), price, "Rabatt"))

    # Fallback: look for pant anywhere in text
    if not any(r.get("Kategori") == "Pant" for r in rows):
        m = _RE_ICA_PANT_ANYWHERE.search(text)
        if m:
            pant_sum = parse_num(m.group(1))
            if abs(pant_sum) > 1e-9:
                rows.append(_make_row(date, "ICA", 9999, "Pant", pant_sum, "Pant"))
        elif debug and "PANT" in text.upper():
            warn(f"{pdf_name}: ICA: 'PANT' finns i text men pant-rad kunde inte matchas", debug, on_warn)

    return rows


# =============================================================================
# COOP PARSER
# =============================================================================

def parse_coop_text(date: str, text: str, categorize_func: Callable,
                    debug: bool = False, on_warn: Optional[Callable] = None) -> List[Dict]:
    """Parse COOP receipt text."""
    if not text:
        return []

    start = _RE_COOP_START.search(text)
    end = _RE_COOP_END.search(text)

    if not start or not end or end.start() <= start.end():
        warn("COOP: kunde inte hitta Org Nr / Total SEK", debug, on_warn)
        return []

    block = text[start.end():end.start()]
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    rows = []

    for i, ln in enumerate(lines, start=1):
        up = ln.upper()

        # Skip multiplier lines and ignored patterns
        if _RE_COOP_MULT.match(ln) or _contains_any(up, _COOP_IGNORE):
            continue

        m = _RE_COOP_PRICE.match(ln)
        if not m:
            continue

        name = m.group(1).strip()
        price = parse_num(m.group(2))

        if abs(price) < 1e-9:
            continue

        # Determine category
        if "PANT" in up:
            cat = "Pant"
        elif price < 0:
            cat = "Rabatt"
        else:
            cat = categorize_func(name)

        rows.append(_make_row(date, "Coop", i, name, price, cat, Mangd=0.0, Enhet="st"))

    return rows


# =============================================================================
# WILLYS PARSER
# =============================================================================

def parse_willys_text(date: str, text: str, categorize_func: Callable,
                      debug: bool = False, on_warn: Optional[Callable] = None) -> List[Dict]:
    """Parse Willys receipt text."""
    if not text:
        return []

    # Find scanning block
    start_m = _RE_WILLYS_START.search(text)
    end_m = _RE_WILLYS_END.search(text)

    block = text
    if start_m and end_m and end_m.start() > start_m.end():
        block = text[start_m.end():end_m.start()]

    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    rows = []
    pending_name = None
    last_item_name = None

    for i, ln in enumerate(lines, start=1):
        up = ln.upper()

        # Skip ignored patterns
        if _contains_any(up, _WILLYS_IGNORE):
            pending_name = None
            continue

        # Discount line
        m = _RE_WILLYS_DISCOUNT.match(ln)
        if m:
            desc = m.group(1).strip()
            price = parse_num(m.group(2))
            if abs(price) > 1e-9:
                rows.append(_make_row(
                    date, "Willys", i,
                    desc or last_item_name or "Rabatt",
                    price,
                    "Rabatt" if price < 0 else "Övrigt",
                    Mangd=0.0, Enhet="st"
                ))
            pending_name = None
            continue

        # Weight line (kg * kr/kg)
        m = _RE_WILLYS_WEIGHT.match(ln)
        if m and pending_name:
            mangd_kg = parse_num(m.group(1))
            price = parse_num(m.group(3))

            if abs(price) > 1e-9:
                name = clean_willys_name(pending_name)
                if name:
                    cat = "Pant" if "PANT" in name.upper() else categorize_func(name)
                    rows.append(_make_row(date, "Willys", i, name, price, cat, Mangd=mangd_kg, Enhet="kg"))
                    last_item_name = name
            pending_name = None
            continue

        # Regular price line
        m = _RE_WILLYS_PRICE.match(ln)
        if m:
            raw_name = m.group(1).strip()
            qty = extract_multiplier_qty(raw_name)
            name = clean_willys_name(raw_name)
            price = parse_num(m.group(2))

            if abs(price) > 1e-9 and name:
                cat = "Pant" if "PANT" in name.upper() else categorize_func(name)
                rows.append(_make_row(date, "Willys", i, name, price, cat, antal=qty, Mangd=0.0, Enhet="st"))
                last_item_name = name
            pending_name = None
            continue

        # Name line for weighted items (next line will be kg*...)
        if _RE_HAS_LETTERS.search(ln) and not _RE_STARTS_DIGIT.search(ln):
            pending_name = clean_willys_name(ln)
        else:
            pending_name = None

    return rows


# =============================================================================
# MAIN DISPATCHER
# =============================================================================

def extract_data(uploaded_file, categorize_func: Callable, debug: bool = False,
                 on_warn: Optional[Callable] = None) -> List[Dict]:
    """Extract data from uploaded PDF receipt."""
    name = getattr(uploaded_file, "name", "<uploaded_file>")

    # Read file
    try:
        data = uploaded_file.read()
        receipt_id = hashlib.sha1(data).hexdigest()[:12]
    except Exception as e:
        warn(f"{name}: kunde inte läsa filen ({e})", debug, on_warn)
        return []

    # Extract text
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
    except Exception as e:
        warn(f"{name}: pdfplumber-fel ({e})", debug, on_warn)
        return []

    if not text or not text.strip():
        warn(f"{name}: ingen text kunde extraheras", debug, on_warn)
        return []

    # Extract date
    dm = _RE_DATE.search(text)
    date = dm.group(1) if dm else "1970-01-01"

    # Detect store and parse
    up = text.upper()
    rows = []

    try:
        if _RE_WILLYS.search(up):
            rows = parse_willys_text(date, text, categorize_func, debug, on_warn)
            if debug and on_warn:
                total = sum(r["Pris"] for r in rows if r.get("Butik") == "Willys")
                on_warn(f"{name}: WILLYS parsad summa = {total:.2f}")
            if not rows:
                warn(f"{name}: WILLYS igenkänd men inga rader hittades", debug, on_warn)

        elif _RE_COOP.search(up):
            rows = parse_coop_text(date, text, categorize_func, debug, on_warn)
            if not rows:
                warn(f"{name}: COOP igenkänd men inga rader hittades", debug, on_warn)

        elif _RE_ICA.search(up):
            rows = parse_ica_text(name, date, text, categorize_func, debug, on_warn)
            if not rows:
                warn(f"{name}: ICA igenkänd men inga rader hittades", debug, on_warn)

        else:
            warn(f"{name}: okänd butik (ingen av WILLYS/COOP/ICA matchade)", debug, on_warn)
            if debug:
                warn(_RE_WHITESPACE.sub(" ", text)[:300], debug, on_warn)
            return []

    except Exception as e:
        warn(f"{name}: parser-crash ({e})", debug, on_warn)
        if debug:
            warn(_RE_WHITESPACE.sub(" ", text)[:300], debug, on_warn)
        return []

    # Add metadata to all rows
    for r in rows:
        r["KvittoID"] = receipt_id
        r["Fil"] = name

    return rows
