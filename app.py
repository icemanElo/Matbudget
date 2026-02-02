"""
app.py - Version 2.0
Förbättringar:
- Borttaget fokusläge (förenklad UX)
- Fixad rabattberäkning (konsekvent mellan flikar)
- Ny flik för att ändra kategorisering på redan kategoriserade varor
- Förbättrat workflow: Statistik → Kvitton → Normalisering → Kategorisering
- Bättre UI med progress indicators och sammanfattningar
- Snabblänkar mellan flikar för smidigare arbetsflöde
"""

import re
import pandas as pd
import plotly.express as px
import streamlit as st
from contextlib import contextmanager
from functools import lru_cache
from typing import Set

from parsers import extract_data
from utils import (
    load_db, save_db, load_rules, save_rules, recategorize_dataframe,
    add_ean_rule, add_name_rule, add_keyword_rule, delete_ean_rule,
    delete_name_rule, delete_keyword_rule, update_keyword_rule,
    suggest_category, categorize_item, canonical_item_name, normalize_name,
    trace_categorization, cleanup_rules, normalize_item_name, is_special_line,
    rule_health_report, apply_rule_health_fixes
)

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(page_title="Matbudget", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"

# Session state (förenklad - borttaget fokusläge)
_SESSION_DEFAULTS = {
    "ui_reset_counter": 0,
    "debug_logs": [],
    "uploader_key": 0,
    "selected_varanorm": None,
    "show_success_toast": None,
}
for key, default in _SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# HELPERS
# =============================================================================
def wkey(name: str) -> str:
    return f"{name}_{st.session_state.ui_reset_counter}"


def bump_ui_reset():
    st.session_state.ui_reset_counter += 1


def add_debug_log(msg: str):
    st.session_state.debug_logs.append(msg)


@contextmanager
def safe_tab(title: str):
    try:
        yield
    except Exception as e:
        st.error(f"Fel i fliken '{title}': {e}")
        add_debug_log(f"TAB ERROR {title}: {e}")


# Pre-compiled regex
_RE_FUZZY_CLEAN = re.compile(r"[^A-ZÅÄÖ0-9]+")
_RE_DIGITS_ONLY = re.compile(r"^[0-9]+$")

# --- Packsize parsing for unit price (kg/l) ---
_RE_PACK_SINGLE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(KG|HG|G|GR|L|DL|CL|ML)\b", re.IGNORECASE)
_RE_PACK_MULTI  = re.compile(r"\b(\d+)\s*[xX]\s*(\d+(?:[.,]\d+)?)\s*(KG|HG|G|GR|L|DL|CL|ML)\b", re.IGNORECASE)

def _to_float(s: str) -> float:
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return 0.0

def extract_pack_qty_base(name: str):
    """
    Returns (qty, base_unit) where base_unit is 'kg' or 'l', or (0, None) if unknown.
    - For weight: normalize to kg
    - For volume: normalize to l
    Supports "500g", "1 kg", "2x500g", "3x33cl", etc.
    """
    if not name or pd.isna(name):
        return 0.0, None

    s = str(name).upper()

    m = _RE_PACK_MULTI.search(s)
    if m:
        mult = int(m.group(1))
        val = _to_float(m.group(2))
        unit = m.group(3).upper()
        val_total = mult * val
    else:
        m = _RE_PACK_SINGLE.search(s)
        if not m:
            return 0.0, None
        val_total = _to_float(m.group(1))
        unit = m.group(2).upper()

    # weight -> kg
    if unit == "KG":
        return val_total, "kg"
    if unit == "HG":
        return val_total * 0.1, "kg"
    if unit in ("G", "GR"):
        return val_total / 1000.0, "kg"

    # volume -> l
    if unit == "L":
        return val_total, "l"
    if unit == "DL":
        return val_total * 0.1, "l"
    if unit == "CL":
        return val_total * 0.01, "l"
    if unit == "ML":
        return val_total / 1000.0, "l"

    return 0.0, None

def format_pack_label(qty: float, base_unit: str) -> str:
    """Human label for pack size from (qty in kg/l)."""
    if not qty or not base_unit:
        return "?"
    if base_unit == "kg":
        # show grams for <1kg if it is near a nice gram value
        g = qty * 1000.0
        if g < 1000:
            return f"{int(round(g))} g"
        # avoid trailing .0
        return f"{qty:g} kg"
    if base_unit == "l":
        ml = qty * 1000.0
        if ml < 1000:
            return f"{int(round(ml))} ml"
        return f"{qty:g} l"
    return "?"



# --- Manual packsize overrides (stored in rules.json) ---
# Keyed by VaraNorm. Values are stored in base units:
#   {"qty": <float in kg or l>, "unit": "kg"|"l"}.
def normalize_pack_input(qty: float, unit: str):
    """Normalize user input into (qty, base_unit) where base_unit is kg or l."""
    if qty is None:
        return 0.0, None
    try:
        q = float(qty)
    except Exception:
        return 0.0, None
    u = (unit or "").strip().lower()

    if q <= 0:
        return 0.0, None

    # weight -> kg
    if u == "kg":
        return q, "kg"
    if u == "g":
        return q / 1000.0, "kg"
    if u == "hg":
        return q * 0.1, "kg"

    # volume -> l
    if u == "l":
        return q, "l"
    if u == "dl":
        return q * 0.1, "l"
    if u == "cl":
        return q * 0.01, "l"
    if u == "ml":
        return q / 1000.0, "l"

    return 0.0, None


def get_pack_override(rules: dict, varanorm: str):
    """Return (qty, unit) in base units (kg/l) for a VaraNorm, or (0, None)."""
    if not rules or not varanorm:
        return 0.0, None
    po = (rules.get("pack_overrides") or {})
    rec = po.get(str(varanorm))
    if isinstance(rec, dict):
        qty = rec.get("qty", 0) or 0
        unit = rec.get("unit")
        try:
            qty = float(qty)
        except Exception:
            qty = 0.0
        unit = (unit or "").strip().lower() or None
        if qty > 0 and unit in ("kg", "l"):
            return qty, unit
    return 0.0, None

_FUZZY_STOP = frozenset({
    "EKO", "EKOLOGISK", "EKOLOGISKT", "KRAV", "VIKT", "KG", "G", "GR", "ML", "CL", "L", "ST", "PACK", "P",
    "LIGHT", "ZERO", "SFRI", "SF", "NS", "MELLAN", "NORMAL", "EXTRA",
    "ORIGINAL", "CLASSIC", "KLASSISK", "MEDIUM", "MILD", "HOT", "HET",
    "UHT", "BARISTA", "GF", "GLUTENFRI", "MIX", "SMOOTH", "CREAMY"
})


@lru_cache(maxsize=2048)
def family_key(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    s = _RE_FUZZY_CLEAN.sub(" ", str(name).upper())
    tokens = [t for t in s.split() if t and t not in _FUZZY_STOP and not _RE_DIGITS_ONLY.fullmatch(t)]
    seen: Set[str] = set()
    unique = [t for t in tokens if not (t in seen or seen.add(t))]
    return " ".join(unique[:2])


def get_discount_sum(dataframe: pd.DataFrame) -> float:
    """Konsekvent beräkning av rabatter - använd överallt."""
    if dataframe.empty or "Pris" not in dataframe.columns:
        return 0.0
    return dataframe[dataframe["Pris"] < 0]["Pris"].sum()


def get_clean_spending(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Returnerar 'ren' data utan pant/rabatt för analys."""
    if dataframe.empty:
        return dataframe
    mask = (dataframe["Pris"] > 0)
    if "Kategori" in dataframe.columns:
        mask &= ~dataframe["Kategori"].isin(["Pant", "Rabatt"])
    return dataframe[mask].copy()


def save_and_rebuild(success_msg: str = "Sparat!"):
    """Standardiserad spara-och-bygg-om-funktion."""
    cleanup_rules()
    df2 = load_db()
    df2 = recategorize_dataframe(df2, force=True)
    if "Vara" in df2.columns:
        df2["VaraNorm"] = df2["Vara"].apply(normalize_item_name)
    save_db(df2)
    bump_ui_reset()
    st.toast(success_msg, icon="✅")


def render_progress_bar(done: int, total: int, label: str = ""):
    """Visar progress för kategorisering/normalisering."""
    if total == 0:
        pct = 100
    else:
        pct = int((done / total) * 100)
    st.progress(pct / 100, text=f"{label}: {done}/{total} ({pct}%)")


# =============================================================================
# LOAD DATA
# =============================================================================
rules = load_rules()
df = load_db()

if not df.empty:
    df["VaraNorm"] = df["Vara"].apply(lambda x: normalize_item_name(x, rules))
    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df["Month"] = df["Datum"].dt.to_period("M").astype(str)
        df["Date"] = df["Datum"].dt.date


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("📤 Ladda upp kvitton")
    files = st.file_uploader(
        "PDF-filer från ICA, Coop, Willys",
        type="pdf",
        accept_multiple_files=True,
        key=f"u_{st.session_state.uploader_key}"
    )

    if st.button("🚀 Processa", key=wkey("proc_btn"), use_container_width=True) and files:
        with st.spinner(f"Bearbetar {len(files)} fil(er)..."):
            new_rows = []
            for f in files:
                rows = extract_data(f, categorize_item, debug=True, on_warn=add_debug_log)
                if rows:
                    new_rows.extend(rows)

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                new_df["Datum"] = pd.to_datetime(new_df["Datum"], errors="coerce")
                new_df["Pris"] = pd.to_numeric(new_df["Pris"], errors="coerce").fillna(0.0)

                merged = pd.concat([df, new_df], ignore_index=True)
                dedup_cols = ["KvittoID", "Rad", "Pris"] if "KvittoID" in merged.columns else ["Datum", "Butik", "Vara", "Pris"]
                merged = merged.drop_duplicates(subset=dedup_cols)
                merged = recategorize_dataframe(merged, force=True)
                save_db(merged)

                st.session_state.uploader_key += 1
                bump_ui_reset()
                st.success(f"✅ {len(new_rows)} rader importerade!")
                st.rerun()
            else:
                st.warning("Inga rader kunde extraheras.")

    # Quick stats i sidebar
    if not df.empty:
        st.divider()
        st.caption("📊 Snabbstatistik")
        
        total_rows = len(df)
        uncat = len(df[df["Kategori"] == "Okategoriserat"]) if "Kategori" in df.columns else 0
        cat_pct = int(((total_rows - uncat) / total_rows) * 100) if total_rows > 0 else 0
        
        st.metric("Totalt", f"{total_rows} rader")
        st.metric("Kategoriserat", f"{cat_pct}%", delta=f"{uncat} kvar" if uncat > 0 else None, delta_color="inverse")
        
        total_spend = get_clean_spending(df)["Pris"].sum() if not df.empty else 0
        st.metric("Total spend", f"{total_spend:,.0f} kr")


# =============================================================================
# MAIN UI
# =============================================================================
st.title("🛒 Matbudget")

if df.empty:
    st.info("👋 Välkommen! Ladda upp kvitton i sidomenyn för att komma igång.")
    st.stop()

# Tabs - nytt ordning för bättre workflow
tabs = st.tabs([
    "📊 Översikt",
    "🧾 Kvitton", 
    "🧠 Normalisering",
    "🏷️ Kategorisera",
    "✏️ Ändra kategori",
    "💰 Prisjämförelse",
    "🗂️ Kategorier",
    "⚙️ Hantera",
    "🪲 Debug",
    "📦 Förpackningar"
])


# =============================================================================
# TAB 1: ÖVERSIKT (förbättrad statistik)
# =============================================================================
with tabs[0]:
    with safe_tab("Översikt"):
        # Workflow status
        total_rows = len(df)
        uncat_count = len(df[df["Kategori"] == "Okategoriserat"])
        unique_products = df["VaraNorm"].nunique() if "VaraNorm" in df.columns else 0
        
        st.subheader("🎯 Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Artikelrader", total_rows)
        c2.metric("Unika produkter", unique_products)
        c3.metric("Kategoriserade", f"{total_rows - uncat_count}", delta=f"{uncat_count} kvar" if uncat_count else "✅ Klart!")
        
        clean_df = get_clean_spending(df)
        discount_total = abs(get_discount_sum(df))
        c4.metric("Totala rabatter", f"{discount_total:,.0f} kr")

        if uncat_count > 0:
            render_progress_bar(total_rows - uncat_count, total_rows, "Kategorisering")

        st.divider()

        # Filter
        st.subheader("📊 Analys")
        cF1, cF2, cF3, cF4 = st.columns([1.2, 1.2, 1.2, 1.4])
        dmin, dmax = df["Datum"].min(), df["Datum"].max()

        with cF1:
            date_from = st.date_input("Från", value=dmin.date() if pd.notna(dmin) else None, key=wkey("f_from"))
        with cF2:
            date_to = st.date_input("Till", value=dmax.date() if pd.notna(dmax) else None, key=wkey("f_to"))
        with cF3:
            stores = ["Alla"] + sorted(df["Butik"].dropna().astype(str).unique().tolist())
            butik = st.selectbox("Butik", stores, key=wkey("f_store"))
        with cF4:
            cats = ["Alla"] + sorted(df["Kategori"].dropna().astype(str).unique().tolist())
            cat = st.selectbox("Kategori", cats, key=wkey("f_cat"))

        # Apply filters
        fdf = df.copy()
        if date_from:
            fdf = fdf[fdf["Datum"].dt.date >= date_from]
        if date_to:
            fdf = fdf[fdf["Datum"].dt.date <= date_to]
        if butik != "Alla":
            fdf = fdf[fdf["Butik"] == butik]
        if cat != "Alla":
            fdf = fdf[fdf["Kategori"] == cat]

        # KPIs med konsekvent rabattberäkning
        filtered_clean = get_clean_spending(fdf)
        filtered_discount = abs(get_discount_sum(fdf))
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Totalt (netto)", f"{fdf['Pris'].sum():,.0f} kr")
        c2.metric("Köp (exkl. pant/rabatt)", f"{filtered_clean['Pris'].sum():,.0f} kr")
        c3.metric("Rabatter", f"{filtered_discount:,.0f} kr")
        c4.metric("Rader", len(fdf))

        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 📈 Per månad")
            m = filtered_clean.groupby("Month")["Pris"].sum().reset_index().sort_values("Month")
            if not m.empty:
                fig = px.bar(m, x="Month", y="Pris", text_auto=".0f")
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("#### 🧩 Per kategori")
            by_cat = filtered_clean.groupby("Kategori")["Pris"].sum().reset_index().sort_values("Pris", ascending=True)
            if not by_cat.empty:
                fig = px.bar(by_cat.tail(10), x="Pris", y="Kategori", orientation="h", text_auto=".0f")
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Top products
        st.write("#### 🏆 Topp 10 produkter")
        top = filtered_clean.groupby("VaraNorm").agg(
            Summa=("Pris", "sum"),
            Antal=("Antal", "sum"),
            Kategori=("Kategori", "first")
        ).sort_values("Summa", ascending=False).head(10).reset_index()
        
        if not top.empty:
            st.dataframe(
                top.style.format({"Summa": "{:,.0f} kr", "Antal": "{:.0f}"}),
                use_container_width=True,
                hide_index=True
            )

        # Rabattlista (ersätter separat rabatt-flik)
        with st.expander("💰 Visa alla rabatter"):
            disc_df = fdf[fdf["Pris"] < 0].copy()
            if disc_df.empty:
                st.info("Inga rabatter i vald period.")
            else:
                disc_df["Belopp"] = disc_df["Pris"].abs()
                st.dataframe(
                    disc_df[["Datum", "Butik", "Vara", "Belopp"]].sort_values("Datum", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )


# =============================================================================
# TAB 2: KVITTON
# =============================================================================
with tabs[1]:
    with safe_tab("Kvitton"):
        st.subheader("🧾 Kvittohistorik")
        
        if "KvittoID" in df.columns and df["KvittoID"].notna().any():
            # Gruppera per kvitto
            kvitto_agg = df.groupby(["KvittoID", "Date", "Butik"]).agg(
                Rader=("Vara", "count"),
                Summa=("Pris", "sum")
            ).reset_index().sort_values("Date", ascending=False)
            
            st.dataframe(
                kvitto_agg.style.format({"Summa": "{:,.2f} kr"}),
                use_container_width=True,
                hide_index=True
            )
            
            # Visa detaljer för valt kvitto
            with st.expander("🔍 Visa kvittoinnehåll"):
                kvitto_ids = kvitto_agg["KvittoID"].tolist()
                sel_kvitto = st.selectbox("Välj kvitto", kvitto_ids, key=wkey("sel_kvitto"))
                if sel_kvitto:
                    detail = df[df["KvittoID"] == sel_kvitto][["Vara", "Antal", "Pris", "Kategori"]]
                    st.dataframe(detail, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df[["Datum", "Butik", "Vara", "Pris", "Kategori"]].head(100), use_container_width=True)


# =============================================================================
# TAB 3: NORMALISERING (före kategorisering i workflow)
# =============================================================================
with tabs[2]:
    with safe_tab("Normalisering"):
        st.subheader("🧠 Normalisering")
        st.caption("Slå ihop varianter av samma produkt (t.ex. 'MJÖLK 3% 1L' och 'MJÖLK EKO 3%' → 'MJÖLK')")

        rules_live = load_rules()
        norm_cfg = rules_live.get("normalization", {"exact": {}, "contains": []})
        if not isinstance(norm_cfg, dict):
            norm_cfg = {"exact": {}, "contains": []}

        # Sökfält prominent
        q = st.text_input("🔎 Sök produkt", placeholder="t.ex. MJÖLK, BREGOTT, FÄRS...", key=wkey("norm_q"))
        
        col1, col2 = st.columns([3, 1])
        with col2:
            show_all = st.checkbox("Visa alla grupper", key=wkey("norm_all"))

        # Bygg arbetsdata
        item_col = "Vara" if "Vara" in df.columns else None
        if not item_col:
            st.error("Ingen Vara-kolumn hittades.")
        else:
            work = get_clean_spending(df)
            work = work[~work[item_col].apply(lambda s: is_special_line(s, rules_live))]

            if work.empty:
                st.success("✅ Ingen data att normalisera.")
            else:
                if "Antal" not in work.columns:
                    work["Antal"] = 1

                work["Base"] = work[item_col].apply(canonical_item_name)
                work["Family"] = work[item_col].apply(family_key)

                grouped = work.groupby(item_col, as_index=False).agg(
                    Vara=(item_col, "first"),
                    Base=("Base", "first"),
                    VaraNorm=("VaraNorm", "first"),
                    Antal=("Antal", "sum"),
                    Family=("Family", "first"),
                )

                qq = (q or "").strip().upper()

                # A) Sökresultat
                if qq:
                    st.write("### Sökresultat")
                    mask = (
                        grouped["Vara"].str.upper().str.contains(qq, na=False) |
                        grouped["Base"].str.upper().str.contains(qq, na=False) |
                        grouped["VaraNorm"].str.upper().str.contains(qq, na=False)
                    )
                    matches = grouped[mask].sort_values("Antal", ascending=False)

                    if matches.empty:
                        st.info(f"Inga produkter matchar '{q}'")
                    else:
                        st.dataframe(matches[["Vara", "VaraNorm", "Antal"]], use_container_width=True, hide_index=True)

                        # Snabb-normalisering
                        with st.form(key=wkey("norm_search_form")):
                            st.write("**Slå ihop valda:**")
                            options = matches["Vara"].tolist()
                            pick = st.multiselect("Välj varianter", options, default=options, key=wkey("norm_pick"))
                            target = st.text_input("Mål (VaraNorm)", value=qq, key=wkey("norm_target"))
                            
                            if st.form_submit_button("✅ Slå ihop", use_container_width=True):
                                if pick and target.strip():
                                    for v in pick:
                                        norm_cfg.setdefault("exact", {})[str(v).upper().strip()] = target.upper().strip()
                                    rules_live["normalization"] = norm_cfg
                                    save_rules(rules_live)
                                    save_and_rebuild(f"Slog ihop {len(pick)} varianter → {target.upper()}")
                                    st.rerun()

                # B) Automatiska förslag
                st.write("### 💡 Förslag på sammanslagningar")
                
                # Baserat på Family (fuzzy)
                fam_groups = grouped[grouped["Family"].str.strip() != ""].groupby("Family").filter(lambda x: x["Vara"].nunique() > 1)
                
                if qq:
                    fam_groups = fam_groups[
                        fam_groups["Vara"].str.upper().str.contains(qq, na=False) |
                        fam_groups["Family"].str.upper().str.contains(qq, na=False)
                    ]

                if fam_groups.empty:
                    st.success("✅ Inga uppenbara dubbletter hittades!")
                else:
                    groups_list = [(fk, g) for fk, g in fam_groups.groupby("Family")]
                    groups_list.sort(key=lambda x: x[1]["Antal"].sum(), reverse=True)
                    
                    max_show = 50 if show_all else 15
                    if len(groups_list) > max_show:
                        st.caption(f"Visar {max_show} av {len(groups_list)} grupper")
                        groups_list = groups_list[:max_show]

                    for fk, g in groups_list:
                        g2 = g.sort_values("Antal", ascending=False)
                        variants = g2["Vara"].unique().tolist()
                        total_antal = int(g2["Antal"].sum())
                        
                        with st.expander(f"🔗 {fk} ({len(variants)} varianter, {total_antal} köp)"):
                            st.dataframe(g2[["Vara", "VaraNorm", "Antal"]], use_container_width=True, hide_index=True)
                            
                            with st.form(key=wkey(f"norm_form_{fk}")):
                                pick = st.multiselect("Välj", variants, default=variants, key=wkey(f"pick_{fk}"))
                                target = st.text_input("Mål", value=fk.upper(), key=wkey(f"tgt_{fk}"))
                                
                                if st.form_submit_button("Slå ihop"):
                                    if pick and target.strip():
                                        for v in pick:
                                            norm_cfg.setdefault("exact", {})[str(v).upper().strip()] = target.upper().strip()
                                        rules_live["normalization"] = norm_cfg
                                        save_rules(rules_live)
                                        save_and_rebuild(f"→ {target.upper()}")
                                        st.rerun()


# =============================================================================
# TAB 4: KATEGORISERA (okategoriserade)
# =============================================================================
with tabs[3]:
    with safe_tab("Kategorisera"):
        st.subheader("🏷️ Kategorisera nya produkter")
        
        uncat = df[df["Kategori"] == "Okategoriserat"].copy()
        
        if uncat.empty:
            all_done = True
            prev_done = st.session_state.get("all_done_prev", False)
            st.success("🎉 Alla produkter är kategoriserade!")
            if not prev_done:
                st.balloons()
            st.session_state["all_done_prev"] = all_done
        else:
            st.session_state["all_done_prev"] = False
            # Progress
            total = len(df)
            done = total - len(uncat)
            render_progress_bar(done, total, "Framsteg")
            
            # Gruppera per VaraNorm
            grp = uncat.groupby("VaraNorm").agg(
                Antal=("Vara", "count"),
                Exempel=("Vara", "first"),
                TotalPris=("Pris", "sum")
            ).sort_values("TotalPris", ascending=False).reset_index()
            
            st.write(f"**{len(grp)} produkter** att kategorisera ({len(uncat)} rader)")
            
            # Lista med snabbkategorisering
            cats = rules.get("categories", [])
            cats_clean = [c for c in cats if c != "Okategoriserat"]
            
            for idx, row in grp.head(20).iterrows():
                vnorm = row["VaraNorm"]
                antal = int(row["Antal"])
                exempel = row["Exempel"]
                
                # Hämta förslag
                sug_cat, reason, _ = suggest_category(exempel, None, df, rules)
                if sug_cat == "Okategoriserat":
                    sug_cat = cats_clean[0] if cats_clean else "Övrigt"
                
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{vnorm}**")
                    st.caption(f"{antal} st • {exempel[:40]}...")
                
                with col2:
                    sel_cat = st.selectbox(
                        "Kategori",
                        cats_clean,
                        index=cats_clean.index(sug_cat) if sug_cat in cats_clean else 0,
                        key=wkey(f"cat_{idx}"),
                        label_visibility="collapsed"
                    )
                
                with col3:
                    if st.button("💾", key=wkey(f"save_{idx}"), help="Spara"):
                        # Skapa keyword-regel baserat på första 2 tokens
                        canon = canonical_item_name(exempel)
                        tokens = canon.split()[:2]
                        keyword = " ".join(tokens) if tokens else canon
                        
                        add_keyword_rule(keyword, sel_cat)
                        save_and_rebuild(f"{vnorm} → {sel_cat}")
                        st.rerun()
                
                st.divider()
            
            if len(grp) > 20:
                st.caption(f"... och {len(grp) - 20} produkter till")


# =============================================================================
# TAB 5: ÄNDRA KATEGORI (ny flik!)
# =============================================================================
with tabs[4]:
    with safe_tab("Ändra kategori"):
        st.subheader("✏️ Ändra kategori på befintliga produkter")
        st.caption("Korrigera felkategoriserade produkter eller ändra ditt kategoriseringssystem.")
        
        # Sök/filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("🔎 Sök produkt", key=wkey("edit_search"))
        with col2:
            filter_cat = st.selectbox("Filtrera kategori", ["Alla"] + rules.get("categories", []), key=wkey("edit_cat"))
        
        # Bygg lista
        edit_df = df.copy()
        if search:
            mask = edit_df["VaraNorm"].str.upper().str.contains(search.upper(), na=False)
            edit_df = edit_df[mask]
        if filter_cat != "Alla":
            edit_df = edit_df[edit_df["Kategori"] == filter_cat]
        
        if edit_df.empty:
            st.info("Inga produkter matchar sökningen.")
        else:
            # Gruppera
            edit_grp = edit_df.groupby(["VaraNorm", "Kategori"]).agg(
                Antal=("Vara", "count"),
                Summa=("Pris", "sum")
            ).reset_index().sort_values("Summa", ascending=False)
            
            cats = [c for c in rules.get("categories", []) if c != "Okategoriserat"]
            
            st.write(f"**{len(edit_grp)} produkter** ({len(edit_df)} rader)")
            
            # Editerbar tabell
            for idx, row in edit_grp.head(30).iterrows():
                vnorm = row["VaraNorm"]
                curr_cat = row["Kategori"]
                antal = int(row["Antal"])
                summa = row["Summa"]
                
                col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 0.5])
                
                with col1:
                    st.write(f"**{vnorm}**")
                
                with col2:
                    st.caption(f"Nu: {curr_cat}")
                    st.caption(f"{antal} st, {summa:,.0f} kr")
                
                with col3:
                    new_cat = st.selectbox(
                        "Ny",
                        cats,
                        index=cats.index(curr_cat) if curr_cat in cats else 0,
                        key=wkey(f"edit_{idx}"),
                        label_visibility="collapsed"
                    )
                
                with col4:
                    if new_cat != curr_cat:
                        if st.button("💾", key=wkey(f"editsave_{idx}")):
                            # Skapa/uppdatera regel
                            add_name_rule(vnorm, new_cat)
                            save_and_rebuild(f"{vnorm}: {curr_cat} → {new_cat}")
                            st.rerun()
                    else:
                        st.write("")  # Placeholder


# =============================================================================
# TAB 6: PRISJÄMFÖRELSE (ny!)
# =============================================================================
with tabs[5]:
    with safe_tab("Prisjämförelse"):
        st.subheader("💰 Prisjämförelse mellan butiker")
        st.caption("Se var du får bäst pris på dina vanliga produkter (ordinarie priser)")
        
        # Filtrera till endast köp (inte rabatter/pant)
        price_df = get_clean_spending(df).copy()
        
        if price_df.empty:
            st.info("Ingen prisdata tillgänglig.")
        else:

            # Beräkna styckpris (pris / antal) + enhetspris (kr/kg eller kr/l när möjligt)
            price_df["Antal"] = price_df["Antal"].replace(0, 1).fillna(1)

            # Bas: per styck/förpackning (nuvarande beteende)
            price_df["Styckpris"] = price_df["Pris"] / price_df["Antal"]

            # Enhetspris: använd Mangd/Enhet om den finns (vikt/volym), annars försök tolka packstorlek i namnet
            def _unit_price(row):
                pris = float(row.get("Pris") or 0.0)
                if pris <= 0:
                    return (None, None)  # ignorera rabatter/pant

                mangd = row.get("Mangd", 0)
                enhet = str(row.get("Enhet") or "").lower().strip()

                if mangd and mangd > 0 and enhet in ("kg", "l"):
                    return (pris / float(mangd), enhet)  # kr/kg eller kr/l

                qty, base_unit = extract_pack_qty_base(row.get("Vara", ""))
                if qty and qty > 0 and base_unit in ("kg", "l"):
                    return (pris / qty, base_unit)

                return (None, None)

            tmp = price_df.apply(_unit_price, axis=1, result_type="expand")
            price_df["Enhetspris"] = tmp[0]
            price_df["Enhetstyp"] = tmp[1]  # 'kg' / 'l' / None

            # Packstorlek (för separat vy 500g vs 1kg etc.)
            def _pack_label(row):
                # Först: explicit vikt/volym från kvittot (om det finns)
                mangd = row.get("Mangd", 0)
                enhet = str(row.get("Enhet") or "").lower().strip()
                if mangd and mangd > 0 and enhet in ("kg", "l"):
                    return format_pack_label(float(mangd), enhet)

                # Annars: tolka från namnet
                qty, base_unit = extract_pack_qty_base(row.get("Vara", ""))
                return format_pack_label(qty, base_unit)

            price_df["Pack"] = price_df.apply(_pack_label, axis=1)

            price_df_all = price_df.copy()

            mode = st.radio(
                "Jämför pris som",
                ["Per styck/förpackning", "Per kg/l (när möjligt)"],
                horizontal=True,
                key=wkey("price_mode")
            )

            if mode == "Per kg/l (när möjligt)":
                # Behåll bara rader där vi faktiskt kan räkna enhetspris
                price_df = price_df[price_df["Enhetspris"].notna()].copy()
                price_df["JämförPrisRad"] = price_df["Enhetspris"]
            else:
                price_df["JämförPrisRad"] = price_df["Styckpris"]

            
            # =================================================================
            # RABATTHANTERING: Identifiera och filtrera bort rabatterade köp
            # =================================================================
            # Strategi: Om ett kvitto har rabattrader, försök koppla dem till produkter
            # Annars: använd statistisk filtrering (ta bort outliers)
            
            # Kolla om vi har rabattinfo på kvittonivå
            if "KvittoID" in df.columns:
                # Hitta kvitton som har rabattrader
                discount_kvitton = df[df["Pris"] < 0]["KvittoID"].unique()
                
                # Markera rader som "möjligen rabatterade" om de är på ett kvitto med rabatt
                price_df["PåRabattKvitto"] = price_df["KvittoID"].isin(discount_kvitton)
            else:
                price_df["PåRabattKvitto"] = False
            
            # Statistisk filtrering: ta bort priser som är >30% under medianen för produkten
            # Detta fångar kampanjpriser även om vi inte har explicit rabattinfo
            def filter_discounted_prices(group):
                if len(group) < 3:
                    return group  # För få datapunkter för statistik
                
                median_price = group["JämförPrisRad"].median()
                # Behåll priser som är inom 30% av medianen (eller högre)
                threshold = median_price * 0.7
                return group[group["JämförPrisRad"] >= threshold]
            
            # Toggle för användaren
            col_filter1, col_filter2 = st.columns([2, 2])
            with col_filter1:
                exclude_discounts = st.checkbox(
                    "🏷️ Exkludera rabatterade priser", 
                    value=True,
                    help="Filtrerar bort priser som verkar vara kampanj/rabatt (>30% under median)",
                    key=wkey("exclude_disc")
                )
            with col_filter2:
                show_discount_info = st.checkbox(
                    "Visa rabattstatistik",
                    value=False,
                    key=wkey("show_disc_info")
                )
            
            if exclude_discounts:
                # Applicera filter per produkt
                price_df_filtered = price_df.groupby("VaraNorm", group_keys=False).apply(filter_discounted_prices)
                
                removed_count = len(price_df) - len(price_df_filtered)
                if removed_count > 0 and show_discount_info:
                    st.info(f"ℹ️ {removed_count} rabatterade köp exkluderade från jämförelsen")
                
                price_df = price_df_filtered
            
            if show_discount_info:
                with st.expander("📊 Rabattstatistik"):
                    on_discount_kvitto = price_df["PåRabattKvitto"].sum()
                    st.write(f"- Köp på kvitton med rabatt: {on_discount_kvitto}")
                    st.write(f"- Totala köprader efter filter: {len(price_df)}")
            
            # -----------------------------------------------------------------
            # 📦 Förpackningsvy: se 500g vs 1kg (både per förpackning och per kg/l)
            # -----------------------------------------------------------------
            # Aggregera per produkt och butik
            price_agg = price_df.groupby(["VaraNorm", "Butik"]).agg(
                MedelPris=("JämförPrisRad", "mean"),
                MedianPris=("JämförPrisRad", "median"),  # Median är mer robust mot outliers
                MinPris=("JämförPrisRad", "min"),
                MaxPris=("JämförPrisRad", "max"),
                AntalKöp=("Vara", "count"),
                SenastKöpt=("Datum", "max")
            ).reset_index()
            
            # Använd median istället för medel för mer robust jämförelse
            price_agg["JämförPris"] = price_agg["MedianPris"]
            
            # Hitta produkter som köpts i flera butiker
            multi_store = price_agg.groupby("VaraNorm").filter(lambda x: x["Butik"].nunique() > 1)
            
            if multi_store.empty:
                st.info("Du behöver köpa samma produkter i olika butiker för att kunna jämföra priser.")
            else:
                # Beräkna besparingspotential
                def calc_savings(group):
                    if len(group) < 2:
                        return None
                    min_price = group["JämförPris"].min()
                    max_price = group["JämförPris"].max()
                    best_store = group.loc[group["JämförPris"].idxmin(), "Butik"]
                    worst_store = group.loc[group["JämförPris"].idxmax(), "Butik"]
                    total_bought = group["AntalKöp"].sum()
                    potential_saving = (max_price - min_price) * total_bought
                    pct_diff = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
                    return pd.Series({
                        "BästButik": best_store,
                        "BästPris": min_price,
                        "DyrastButik": worst_store,
                        "DyrastPris": max_price,
                        "Skillnad": max_price - min_price,
                        "SkillnadPct": pct_diff,
                        "TotaltKöpt": total_bought,
                        "PotentiellBesparing": potential_saving
                    })
                
                savings = multi_store.groupby("VaraNorm").apply(calc_savings).reset_index()
                savings = savings.dropna().sort_values("PotentiellBesparing", ascending=False)
                
                # Sammanfattning
                total_potential = savings["PotentiellBesparing"].sum()
                st.metric(
                    "💡 Total besparingspotential", 
                    f"{total_potential:,.0f} kr",
                    help="Om du alltid handlat varje produkt i den billigaste butiken (baserat på medianpris)"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Produkter att jämföra", len(savings))
                with col2:
                    avg_diff = savings["SkillnadPct"].mean()
                    st.metric("Genomsnittlig prisskillnad", f"{avg_diff:.0f}%")
                
                st.divider()
                
                # Filter
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    search_price = st.text_input("🔎 Sök produkt", key=wkey("price_search"))
                with col2:
                    min_saving = st.number_input("Min besparing (kr)", value=0, step=10, key=wkey("min_save"))
                with col3:
                    min_diff_pct = st.number_input("Min skillnad (%)", value=0, step=5, key=wkey("min_diff"))
                
                # Filtrera
                filtered = savings.copy()
                if search_price:
                    filtered = filtered[filtered["VaraNorm"].str.upper().str.contains(search_price.upper(), na=False)]
                if min_saving > 0:
                    filtered = filtered[filtered["PotentiellBesparing"] >= min_saving]
                if min_diff_pct > 0:
                    filtered = filtered[filtered["SkillnadPct"] >= min_diff_pct]
                
                if filtered.empty:
                    st.info("Inga produkter matchar filtret.")
                else:
                    # Topp besparingar
                    st.write("### 🏆 Störst besparingspotential")
                    
                    for idx, row in filtered.head(20).iterrows():
                        vnorm = row["VaraNorm"]
                        best = row["BästButik"]
                        worst = row["DyrastButik"]
                        best_price = row["BästPris"]
                        worst_price = row["DyrastPris"]
                        diff_pct = row["SkillnadPct"]
                        potential = row["PotentiellBesparing"]
                        
                        with st.container():
                            col1, col2, col3 = st.columns([3, 2, 1])
                            
                            with col1:
                                st.write(f"**{vnorm}**")
                                st.caption(f"Köpt {int(row['TotaltKöpt'])} gånger totalt")
                            
                            with col2:
                                st.write(f"✅ **{best}**: {best_price:.2f} kr")
                                st.write(f"❌ {worst}: {worst_price:.2f} kr")
                            
                            with col3:
                                st.metric(
                                    "Besparing",
                                    f"{potential:.0f} kr",
                                    delta=f"-{diff_pct:.0f}%",
                                    delta_color="normal"
                                )
                            
                            st.divider()
                    
                    # Detaljerad tabell
                    with st.expander("📊 Visa detaljerad tabell"):
                        display_df = filtered[[
                            "VaraNorm", "BästButik", "BästPris", "DyrastButik", "DyrastPris", 
                            "SkillnadPct", "PotentiellBesparing"
                        ]].copy()
                        display_df.columns = ["Produkt", "Bäst butik", "Bäst pris", "Dyrast butik", "Dyrast pris", "Skillnad %", "Besparing kr"]
                        
                        st.dataframe(
                            display_df.style.format({
                                "Bäst pris": "{:.2f}",
                                "Dyrast pris": "{:.2f}",
                                "Skillnad %": "{:.0f}%",
                                "Besparing kr": "{:.0f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Butiksjämförelse
                    with st.expander("🏪 Vilken butik är billigast totalt?"):
                        # Räkna hur ofta varje butik är billigast
                        best_counts = filtered["BästButik"].value_counts().reset_index()
                        best_counts.columns = ["Butik", "Antal produkter billigast"]
                        
                        worst_counts = filtered["DyrastButik"].value_counts().reset_index()
                        worst_counts.columns = ["Butik", "Antal produkter dyrast"]
                        
                        comparison = best_counts.merge(worst_counts, on="Butik", how="outer").fillna(0)
                        comparison["Netto"] = comparison["Antal produkter billigast"] - comparison["Antal produkter dyrast"]
                        comparison = comparison.sort_values("Netto", ascending=False)
                        
                        st.dataframe(comparison, use_container_width=True, hide_index=True)
                        
                        # Visualisering
                        if len(comparison) > 1:
                            fig = px.bar(
                                comparison, 
                                x="Butik", 
                                y=["Antal produkter billigast", "Antal produkter dyrast"],
                                barmode="group",
                                color_discrete_map={
                                    "Antal produkter billigast": "#2ecc71",
                                    "Antal produkter dyrast": "#e74c3c"
                                }
                            )
                            fig.update_layout(height=300, legend_title="")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Produktdetaljer
                    with st.expander("🔍 Detaljerad prishistorik per produkt"):
                        sel_product = st.selectbox(
                            "Välj produkt",
                            filtered["VaraNorm"].tolist(),
                            key=wkey("price_detail_product")
                        )
                        
                        if sel_product:
                            product_data = price_agg[price_agg["VaraNorm"] == sel_product].sort_values("JämförPris")
                            
                            st.write(f"**Prisöversikt för {sel_product}** (ordinarie priser):")
                            
                            for _, prow in product_data.iterrows():
                                butik = prow["Butik"]
                                median = prow["MedianPris"]
                                minp = prow["MinPris"]
                                maxp = prow["MaxPris"]
                                antal = int(prow["AntalKöp"])
                                
                                col1, col2, col3 = st.columns([2, 2, 1])
                                with col1:
                                    st.write(f"**{butik}**")
                                with col2:
                                    st.write(f"Median: **{median:.2f} kr** (spann: {minp:.2f} - {maxp:.2f})")
                                with col3:
                                    st.caption(f"{antal} köp")
                            
                            # Prishistorik över tid - visa ALLA priser (även filtrerade) med markering
                            all_product_history = get_clean_spending(df)[get_clean_spending(df)["VaraNorm"] == sel_product].copy()
                            if not all_product_history.empty:
                                all_product_history["Antal"] = all_product_history["Antal"].replace(0, 1).fillna(1)
                                all_product_history["Styckpris"] = all_product_history["Pris"] / all_product_history["Antal"]
                                
                                # Markera vilka som är "rabatterade"
                                median_price = all_product_history["Styckpris"].median()
                                threshold = median_price * 0.7
                                all_product_history["Pristyp"] = all_product_history["Styckpris"].apply(
                                    lambda x: "🏷️ Rabatterat" if x < threshold else "Ordinarie"
                                )
                                
                                if len(all_product_history) > 1:
                                    st.write("**Prishistorik över tid:**")
                                    fig = px.scatter(
                                        all_product_history,
                                        x="Datum",
                                        y="Styckpris",
                                        color="Butik",
                                        symbol="Pristyp",
                                        title=f"Alla priser: {sel_product}",
                                        hover_data=["Vara", "Pris"]
                                    )
                                    # Lägg till linje för median
                                    fig.add_hline(
                                        y=median_price, 
                                        line_dash="dash", 
                                        line_color="gray",
                                        annotation_text=f"Median: {median_price:.2f} kr"
                                    )
                                    fig.add_hline(
                                        y=threshold, 
                                        line_dash="dot", 
                                        line_color="red",
                                        annotation_text="Rabattgräns (70%)"
                                    )
                                    fig.update_layout(height=350)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Visa antal rabatterade
                                    n_discounted = (all_product_history["Pristyp"] == "🏷️ Rabatterat").sum()
                                    if n_discounted > 0:
                                        st.caption(f"ℹ️ {n_discounted} av {len(all_product_history)} köp verkar vara rabatterade")


# =============================================================================
# TAB 7: KATEGORIER
# =============================================================================
with tabs[6]:
    with safe_tab("Kategorier"):
        st.subheader("🗂️ Hantera kategorier")
        
        current_cats = rules.get("categories", [])
        
        # Visa kategorier med statistik
        cat_stats = df.groupby("Kategori").agg(
            Rader=("Vara", "count"),
            Summa=("Pris", "sum")
        ).reset_index().sort_values("Summa", ascending=False)
        
        st.dataframe(
            cat_stats.style.format({"Summa": "{:,.0f} kr"}),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # Lägg till ny
        with st.form("add_cat_form"):
            new_cat = st.text_input("Ny kategori")
            if st.form_submit_button("➕ Lägg till"):
                if new_cat and new_cat not in current_cats:
                    rules["categories"].append(new_cat)
                    rules["categories"].sort()
                    save_rules(rules)
                    st.rerun()
        
        # Ta bort (med varning)
        with st.expander("🗑️ Ta bort kategori"):
            del_cat = st.selectbox("Välj kategori att ta bort", current_cats, key=wkey("del_cat"))
            count = len(df[df["Kategori"] == del_cat]) if "Kategori" in df.columns else 0
            if count > 0:
                st.warning(f"⚠️ {count} rader använder denna kategori!")
            if st.button("Ta bort", type="secondary"):
                if del_cat in rules["categories"]:
                    rules["categories"].remove(del_cat)
                    save_rules(rules)
                    st.rerun()


# =============================================================================
# TAB 8: HANTERA
# =============================================================================
with tabs[7]:
    with safe_tab("Hantera"):
        st.subheader("⚙️ Datahantering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🩺 Regelhälsa")
            rep = rule_health_report(rules, df=df)
            
            c1, c2 = st.columns(2)
            c1.metric("EAN-regler", rep["counts"].get("ean_rules", 0))
            c2.metric("Keyword-regler", rep["counts"].get("keyword_rules", 0))
            
            issues = rep.get("issues", {})
            kw_dups = issues.get("keyword_duplicates", {})
            
            if kw_dups:
                st.warning(f"⚠️ {len(kw_dups)} keyword-dubbletter")
                if st.button("🧹 Rensa dubbletter"):
                    cleanup_rules()
                    st.rerun()
            else:
                st.success("✅ Inga problem hittade")
        
        with col2:
            st.write("### 🗑️ Rensa data")
            
            with st.expander("Rensa per butik"):
                stores = sorted(df["Butik"].dropna().unique().tolist())
                sel_store = st.selectbox("Butik", stores, key=wkey("clr_store"))
                n = len(df[df["Butik"] == sel_store])
                st.warning(f"Tar bort {n} rader")
                
                if st.button("Rensa butik", key=wkey("clr_btn")):
                    df2 = load_db()
                    df2 = df2[df2["Butik"] != sel_store]
                    save_db(df2)
                    st.rerun()
            
            with st.expander("⚠️ Rensa ALL data"):
                confirm = st.text_input("Skriv RADERA", key=wkey("clr_confirm"))
                if st.button("Rensa allt", disabled=confirm != "RADERA", type="primary"):
                    save_db(df.iloc[0:0])
                    st.rerun()
        
        # Visa regler
        st.divider()
        with st.expander("📋 Visa alla regler"):
            st.json(rules)


# =============================================================================
# TAB 9: DEBUG
# =============================================================================
with tabs[8]:
    st.subheader("🪲 Debug")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Logg")
        st.text_area("", "\n".join(st.session_state.debug_logs[-50:]), height=300)
    
    with col2:
        st.write("### Data preview")
        st.dataframe(df.head(20), use_container_width=True)
    
    st.write("### Session state")
    st.json({k: str(v)[:100] for k, v in st.session_state.items()})


with tabs[9]:
    with safe_tab("Förpackningar"):
        st.subheader("📦 Förpackningar och packstorlekar")
        st.caption("Separat vy som grupperar på VaraNorm + packstorlek (t.ex. 500 g vs 1 kg, 33 cl vs 1 l). Påverkar inget annat.")

        base = get_clean_spending(df).copy()

        rules_pack = load_rules()
        pack_overrides = rules_pack.get("pack_overrides", {})

        if base.empty:
            st.info("Ingen prisdata tillgänglig.")
        else:
            # Beräkna styckpris (pris / antal) + enhetspris (kr/kg eller kr/l när möjligt)
            base["Antal"] = base["Antal"].replace(0, 1).fillna(1)
            base["Styckpris"] = base["Pris"] / base["Antal"]

            def _unit_price(row):
                pris = float(row.get("Pris") or 0.0)
                if pris <= 0:
                    return (None, None)

                # 0) Manual override (if provided for this VaraNorm)
                ov_qty, ov_unit = get_pack_override(rules_pack, row.get("VaraNorm", ""))
                if ov_qty and ov_unit in ("kg", "l"):
                    return (pris / float(ov_qty), ov_unit)

                # 1) Prefer explicit receipt weight/volume
                mangd = row.get("Mangd", 0)
                enhet = str(row.get("Enhet") or "").lower().strip()
                if mangd and mangd > 0 and enhet in ("kg", "l"):
                    return (pris / float(mangd), enhet)

                # 2) Fallback: parse from product name
                qty, base_unit = extract_pack_qty_base(row.get("Vara", ""))
                if qty and qty > 0 and base_unit in ("kg", "l"):
                    return (pris / qty, base_unit)

                return (None, None)


            tmp = base.apply(_unit_price, axis=1, result_type="expand")
            base["Enhetspris"] = tmp[0]
            base["Enhetstyp"] = tmp[1]

            def _pack_label(row):
                # 0) Manual override (if provided for this VaraNorm)
                ov_qty, ov_unit = get_pack_override(rules_pack, row.get("VaraNorm", ""))
                if ov_qty and ov_unit in ("kg", "l"):
                    return format_pack_label(float(ov_qty), ov_unit)

                # 1) Prefer explicit receipt weight/volume
                mangd = row.get("Mangd", 0)
                enhet = str(row.get("Enhet") or "").lower().strip()
                if mangd and mangd > 0 and enhet in ("kg", "l"):
                    return format_pack_label(float(mangd), enhet)

                # 2) Fallback: parse from product name
                qty, base_unit = extract_pack_qty_base(row.get("Vara", ""))
                return format_pack_label(qty, base_unit)

            base["Pack"] = base.apply(_pack_label, axis=1)

            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                pack_search = st.text_input("🔎 Sök VaraNorm", key=wkey("pack_tab_search"))
            with c2:
                price_mode = st.radio(
                    "Visa priser som",
                    ["Per kg/l (när möjligt)", "Per styck/förpackning"],
                    horizontal=False,
                    key=wkey("pack_tab_mode")
                )
            with c3:
                only_multi = st.checkbox("Visa bara flera storlekar", value=False, key=wkey("pack_only_multi"))

            show_unknown = st.checkbox("Visa okänd packstorlek (?)", value=True, key=wkey("pack_show_unknown"))

            view = base.copy()
            if pack_search:
                view = view[view["VaraNorm"].str.upper().str.contains(pack_search.upper(), na=False)]

            if not show_unknown:
                view = view[view["Pack"] != "?"].copy()

            if price_mode == "Per kg/l (när möjligt)":
                view = view[view["Enhetspris"].notna()].copy()
                view["PackPris"] = view["Enhetspris"]
                view["PackPrisTyp"] = view["Enhetstyp"]  # kg/l
            else:
                view["PackPris"] = view["Styckpris"]
                view["PackPrisTyp"] = "st"

            if view.empty:
                st.info("Inga rader matchar dina filter (eller saknar enhetspris i 'Per kg/l'-läge).")
            else:
                pack_agg = view.groupby(["VaraNorm", "Pack", "Butik", "PackPrisTyp"]).agg(
                    MedianPris=("PackPris", "median"),
                    MedelPris=("PackPris", "mean"),
                    MinPris=("PackPris", "min"),
                    MaxPris=("PackPris", "max"),
                    AntalKöp=("Vara", "count"),
                    SenastKöpt=("Datum", "max")
                ).reset_index()

                if only_multi:
                    pack_agg = pack_agg.groupby("VaraNorm").filter(lambda x: x["Pack"].nunique() > 1)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Rader i vyn", len(pack_agg))
                mc2.metric("Produkter i vyn", pack_agg["VaraNorm"].nunique() if not pack_agg.empty else 0)
                mc3.metric("Unika pack", pack_agg["Pack"].nunique() if not pack_agg.empty else 0)

                st.write("### 📋 Packstorlekar per produkt och butik")
                st.dataframe(
                    pack_agg.sort_values(["VaraNorm", "Pack", "Butik"]),
                    use_container_width=True,
                    hide_index=True
                )
        st.divider()
        st.subheader("✍️ Sätt packstorlek manuellt (när inget hittas)")
        st.caption("Detta används bara för pack-/enhetsprisvyer. Normalisering/kategorisering påverkas inte.")
        # Lista produkter med okänd packstorlek
        unknown_df = base[base["Pack"] == "?"].copy()
        if unknown_df.empty:
            st.info("Inga produkter med okänd packstorlek just nu.")
        else:
            cM1, cM2 = st.columns([2, 1])
        with cM1:
            vc = unknown_df["VaraNorm"].fillna("?").value_counts()
            unknown_options = vc.reset_index()
            # Pandas-versioner skiljer sig lite i kolumnnamn här; sätt dem explicit.
            unknown_options.columns = ["VaraNorm", "Antal rader"]
            st.write("**Okända packstorlekar (topplista)**")
            st.dataframe(unknown_options.head(25), use_container_width=True, hide_index=True)

        with cM2:
            varanorm_list = unknown_options["VaraNorm"].tolist()
            sel = st.selectbox("Välj VaraNorm att rätta", varanorm_list, key=wkey("pack_override_sel"))

            cur_qty, cur_unit = get_pack_override(rules_pack, sel)
            if cur_qty and cur_unit:
                st.info(f"Nuvarande override: {format_pack_label(cur_qty, cur_unit)}")
            else:
                st.info("Ingen override satt ännu.")

            # Visa exempel på råa namn som saknar pack
            examples = (
                unknown_df[unknown_df["VaraNorm"] == sel]["Vara"]
                .dropna().astype(str).value_counts().head(6).index.tolist()
            )
            if examples:
                st.write("**Exempel (råa namn):**")
                for ex in examples:
                    st.write(f"- {ex}")

            qty_in = st.number_input("Mängd", min_value=0.0, value=float(cur_qty) if cur_qty else 0.0, step=0.1, key=wkey("pack_override_qty"))
            unit_in = st.selectbox("Enhet", ["g", "hg", "kg", "ml", "cl", "dl", "l"], index=2, key=wkey("pack_override_unit"))

            nqty, nunit = normalize_pack_input(qty_in, unit_in)
            if nqty and nunit:
                st.write(f"➡️ Sparas som: **{format_pack_label(nqty, nunit)}**")
            else:
                st.warning("Fyll i en positiv mängd och en giltig enhet.")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Spara override", use_container_width=True, key=wkey("pack_override_save")):
                    if not (nqty and nunit):
                        st.error("Kan inte spara: ange en positiv mängd och giltig enhet.")
                    else:
                        rules_pack.setdefault("pack_overrides", {})
                        rules_pack["pack_overrides"][str(sel)] = {"qty": float(nqty), "unit": str(nunit)}
                        save_rules(rules_pack)
                        st.success("Override sparad ✅")
                        st.rerun()
            with b2:
                if st.button("🗑️ Ta bort override", use_container_width=True, key=wkey("pack_override_del")):
                    po = rules_pack.get("pack_overrides", {}) or {}
                    if str(sel) in po:
                        del po[str(sel)]
                        rules_pack["pack_overrides"] = po
                        save_rules(rules_pack)
                        st.success("Override borttagen ✅")
                        st.rerun()
                    else:
                        st.info("Ingen override att ta bort.")

