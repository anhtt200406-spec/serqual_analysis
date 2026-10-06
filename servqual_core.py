"""SERVQUAL Core Utilities Module
=================================
Module cung cap cac ham xu ly, lam sach va tinh toan chi so SERVQUAL cho WinMart.

Ham chinh:
- slugify              : Chuan hoa chuoi khong dau, snake_case.
- rename_column        : Chuyen tieu de Google Form thanh ma bien chuan.
- normalize_value      : Chuan hoa gia tri danh muc (Demographics).
- clean_dataset        : Pipeline loc va lam sach du lieu theo chuan nghien cuu.
- calculate_servqual_scores : Tinh P, E, Gap = P - E theo bien va theo chieu.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd

# ===========================================================================
# CONSTANTS
# ===========================================================================

DIMENSIONS: Dict[int, str] = {
    1: "tangibles",
    2: "reliability",
    3: "responsiveness",
    4: "assurance",
    5: "empathy",
}

DIMENSION_LABELS: Dict[str, str] = {
    "tangibles":       "Phuong tien huu hinh",
    "reliability":     "Do tin cay",
    "responsiveness":  "Kha nang dap ung",
    "assurance":       "Su dam bao",
    "empathy":         "Su thau cam",
}

ITEM_RE = re.compile(r"^\s*(\d+)\.(\d+)\s")

DEMO_KEYWORDS: List[Tuple[str, str]] = [
    ("dau thoi gian",           "timestamp"),
    ("gioi tinh",               "gender"),
    ("sinh vien nam may",       "study_year"),
    ("cu tru",                  "residence"),
    ("sinh hoat phi",           "monthly_budget"),
    ("tan suat",                "shop_frequency"),
    ("nhan duoc bai khao sat",  "survey_source"),
]

VALUE_MAPS: Dict[str, Dict[str, str]] = {
    "gender": {
        "nam": "male",
        "nu":  "female",
    },
    "residence": {
        "ky tuc xa truong":    "dormitory",
        "nha tro o cung ban":  "rental_shared",
        "nha tro o mot minh":  "rental_alone",
        "song cung gia dinh":  "with_family",
    },
    "monthly_budget": {
        "duoi 3 trieu vnd": "under_3m",
        "tu 3 7 trieu vnd": "3m_to_7m",
        "tren 7 trieu vnd": "over_7m",
    },
    "shop_frequency": {
        "hang ngay":                    "daily",
        "2 3 lan tuan":                 "2_3_per_week",
        "1 lan tuan":                   "weekly",
        "thinh thoang duoi 2 lan thang":"occasionally",
    },
}


# ===========================================================================
# HELPERS
# ===========================================================================

def slugify(text: str) -> str:
    """Bo dau tieng Viet, loai ky tu dac biet, chuyen thanh snake_case."""
    if text is None:
        return ""
    text = str(text).strip().replace("d", "d").replace("D", "D")
    text = str(text).strip().replace("\u0111", "d").replace("\u0110", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def rename_column(col: str) -> str:
    """Anh xa tieu de cot Google Forms sang ten bien tieng Anh chuan (snake_case)."""
    m = ITEM_RE.match(str(col))
    if m:
        block = int(m.group(1))
        idx   = int(m.group(2))
        kind  = "p" if block <= 5 else "e"          # 1-5: Perception, 6-10: Expectation
        dim   = DIMENSIONS[block if block <= 5 else block - 5]
        return f"{dim}_{idx}_{kind}"
    key = slugify(col).replace("_", " ")
    for kw, name in DEMO_KEYWORDS:
        if kw in key:
            return name
    return slugify(col)


def normalize_value(col: str, value):
    """Chuan hoa gia tri phan loai thanh nhan tieng Anh khong dau."""
    if pd.isna(value):
        return value
    key = slugify(value).replace("_", " ")
    if col == "study_year":
        m = re.search(r"nam\s*(\d+)", key)
        return f"year_{m.group(1)}" if m else "not_university_student"
    mapped = VALUE_MAPS.get(col, {}).get(key)
    return mapped if mapped else slugify(value)


# ===========================================================================
# SORT KEY (dung noi bo)
# ===========================================================================

def _item_sort_key(col: str):
    """Sap xep cot item theo dimension roi theo so thu tu."""
    dim, idx, _ = col.rsplit("_", 2)
    return (list(DIMENSIONS.values()).index(dim), int(idx))


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def clean_dataset(
    df_raw: pd.DataFrame,
    drop_straightline: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lam sach va kiem tra chat luong du lieu SERVQUAL.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Du lieu goc xuat tu Google Forms.
    drop_straightline : bool
        Neu True, loai cac phieu chon cung 1 muc diem cho ca 32 cau.

    Returns
    -------
    clean_df : pd.DataFrame
        Du lieu da lam sach (chi giu phieu hop le).
    dropped_df : pd.DataFrame
        Cac phieu bi loai kem cot drop_reason.
    stats : dict
        Bao cao tom tat (n_raw, n_clean, n_dropped, drop_breakdown, valid_rate).
    """
    df = df_raw.copy()
    n_raw = len(df)

    # 1. Doi ten cot
    df.columns = [rename_column(c) for c in df.columns]

    p_cols = [c for c in df.columns if c.endswith("_p")]
    e_cols = [c for c in df.columns if c.endswith("_e")]
    item_cols = p_cols + e_cols
    demo_cols = [c for c in df.columns if c not in item_cols]

    # 2. Sap xep thu tu cot: demographics -> P (dim1..5) -> E (dim1..5)
    p_sorted = sorted(p_cols, key=_item_sort_key)
    e_sorted = sorted(e_cols, key=_item_sort_key)
    df = df[demo_cols + p_sorted + e_sorted]
    item_cols = p_sorted + e_sorted

    # 3. Timestamp: so serial Excel -> datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="D", origin="1899-12-30", errors="coerce"
        )

    # 4. Chuan hoa gia tri cac cot phan loai
    for col in demo_cols:
        if col != "timestamp":
            df[col] = df[col].map(lambda v, c=col: normalize_value(c, v))

    # 5. Kiem tra thang do Likert (1-5); ngoai mien nay -> NaN
    df[item_cols] = df[item_cols].apply(pd.to_numeric, errors="coerce")
    df[item_cols] = df[item_cols].where(df[item_cols].isin([1, 2, 3, 4, 5]))

    # 6. Danh so thu tu respondent_id (giu nguyen thu tu goc)
    df.insert(0, "respondent_id", range(1, len(df) + 1))

    # 7. Danh dau ly do loai (uu tien tu tren xuong)
    reason = pd.Series("", index=df.index)

    def flag(mask, label):
        nonlocal reason
        reason = reason.mask(mask & (reason == ""), label)

    if "study_year" in df.columns:
        flag(df["study_year"].eq("not_university_student"), "not_university_student")
    flag(df[item_cols].isna().any(axis=1), "missing_or_invalid_item")
    flag(df[item_cols].eq(5).all(axis=1), "all_items_5")
    if drop_straightline:
        flag(df[item_cols].nunique(axis=1).eq(1), "straightlining")

    dropped_df = df[reason != ""].assign(drop_reason=reason[reason != ""]).copy()
    clean_df   = df[reason == ""].reset_index(drop=True).copy()
    clean_df[item_cols] = clean_df[item_cols].astype(int)

    stats = {
        "n_raw":           n_raw,
        "n_clean":         len(clean_df),
        "n_dropped":       len(dropped_df),
        "drop_breakdown":  dropped_df["drop_reason"].value_counts().to_dict() if len(dropped_df) else {},
        "valid_rate":      len(clean_df) / n_raw if n_raw > 0 else 0,
    }

    return clean_df, dropped_df, stats


# ===========================================================================
# SERVQUAL SCORING
# ===========================================================================

def calculate_servqual_scores(
    clean_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tinh toan diem trung binh P, E va Gap = P - E theo bien va theo chieu.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Du lieu sach (dau ra cua clean_dataset).

    Returns
    -------
    df_item_scores : pd.DataFrame
        Bang ket qua cap do bien (moi dong = 1 cap P/E).
    df_dim_scores : pd.DataFrame
        Bang ket qua cap do chieu SERVQUAL (Tangibles, Reliability...).
    """
    p_cols = [c for c in clean_df.columns if c.endswith("_p")]

    item_records = []
    for p_col in p_cols:
        dim, idx, _ = p_col.rsplit("_", 2)
        e_col = f"{dim}_{idx}_e"

        p_series   = clean_df[p_col]
        e_series   = clean_df[e_col]
        gap_series = p_series - e_series

        item_records.append({
            "dimension":    dim,
            "dimension_vn": DIMENSION_LABELS.get(dim, dim),
            "item_code":    f"{dim}_{idx}",
            "p_col":        p_col,
            "e_col":        e_col,
            "p_mean":       round(p_series.mean(), 4),
            "p_std":        round(p_series.std(),  4),
            "e_mean":       round(e_series.mean(), 4),
            "e_std":        round(e_series.std(),  4),
            "gap_mean":     round(gap_series.mean(), 4),
            "gap_std":      round(gap_series.std(),  4),
        })

    df_items = pd.DataFrame(item_records)
    df_items["gap_rank"] = df_items["gap_mean"].rank(ascending=True).astype(int)

    # Cap do chieu SERVQUAL
    dim_records = []
    for dim_code, dim_vn in DIMENSION_LABELS.items():
        dim_p_cols = [c for c in p_cols if c.startswith(f"{dim_code}_")]
        dim_e_cols = [c.replace("_p", "_e") for c in dim_p_cols]
        if not dim_p_cols:
            continue
        p_mean   = round(clean_df[dim_p_cols].values.mean(), 4)
        e_mean   = round(clean_df[dim_e_cols].values.mean(), 4)
        gap_mean = round(p_mean - e_mean, 4)
        dim_records.append({
            "dimension":    dim_code,
            "dimension_vn": dim_vn,
            "num_items":    len(dim_p_cols),
            "p_mean":       p_mean,
            "e_mean":       e_mean,
            "gap_mean":     gap_mean,
        })

    df_dims = pd.DataFrame(dim_records)
    df_dims["gap_rank"] = df_dims["gap_mean"].rank(ascending=True).astype(int)

    return df_items, df_dims
