#!/usr/bin/env python3
"""Lam sach & chuan hoa du lieu SERVQUAL WinMart.

Dau vao : data/data_servqual.xlsx  (file xlsx xuat tu Google Forms)
Dau ra  : data/<ten>_clean.csv     (du lieu sach, cot tieng Anh khong dau)
          data/<ten>_dropped.csv   (cac phieu bi loai + ly do)
          [Thu muc data/ da duoc gitignore]

Quy uoc ten cot muc hoi (item):
    <dimension>_<so thu tu>_<p|e>
    1.1 -> tangibles_1_p   (P = Perception / Cam nhan)
    6.1 -> tangibles_1_e   (E = Expectation / Ky vong)

Cach dung:
    python clean.py data/data_servqual.xlsx
    python clean.py data/data_servqual.xlsx --drop-straightline
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

DIMENSIONS = {1: "tangibles", 2: "reliability", 3: "responsiveness",
              4: "assurance", 5: "empathy"}

# ---------- helpers ----------

def slugify(text) -> str:
    """Bo dau tieng Viet, chuyen thanh snake_case."""
    text = str(text).strip().replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


ITEM_RE = re.compile(r"^\s*(\d+)\.(\d+)\s")          # "1.1 Khong gian..."
DEMO_KEYWORDS = [                                    # tu khoa (da bo dau) -> ten cot
    ("dau thoi gian", "timestamp"),
    ("gioi tinh", "gender"),
    ("sinh vien nam may", "study_year"),
    ("cu tru", "residence"),
    ("sinh hoat phi", "monthly_budget"),
    ("tan suat", "shop_frequency"),
    ("nhan duoc bai khao sat", "survey_source"),
]


def rename_column(col: str) -> str:
    m = ITEM_RE.match(str(col))
    if m:
        block, idx = int(m.group(1)), int(m.group(2))
        kind = "p" if block <= 5 else "e"             # 1-5: Perception, 6-10: Expectation
        dim = DIMENSIONS[block if block <= 5 else block - 5]
        return f"{dim}_{idx}_{kind}"
    key = slugify(col).replace("_", " ")
    for kw, name in DEMO_KEYWORDS:
        if kw in key:
            return name
    return slugify(col)


# Gia tri phan loai -> nhan tieng Anh khong dau (khong co trong map thi tu slugify)
VALUE_MAPS = {
    "gender": {"nam": "male", "nu": "female"},
    "residence": {
        "ky tuc xa truong": "dormitory",
        "nha tro o cung ban": "rental_shared",
        "nha tro o mot minh": "rental_alone",
        "song cung gia dinh": "with_family",
    },
    "monthly_budget": {
        "duoi 3 trieu vnd": "under_3m",
        "tu 3 7 trieu vnd": "3m_to_7m",
        "tren 7 trieu vnd": "over_7m",
    },
    "shop_frequency": {
        "hang ngay": "daily",
        "2 3 lan tuan": "2_3_per_week",
        "1 lan tuan": "weekly",
        "thinh thoang duoi 2 lan thang": "occasionally",
    },
}


def normalize_value(col: str, value):
    if pd.isna(value):
        return value
    key = slugify(value).replace("_", " ")
    if col == "study_year":                           # "Sinh vien Nam 3" -> year_3
        m = re.search(r"nam\s*(\d+)", key)
        return f"year_{m.group(1)}" if m else "not_university_student"
    mapped = VALUE_MAPS.get(col, {}).get(key)
    return mapped if mapped else slugify(value)


# ---------- main pipeline ----------

def clean(path: Path, drop_straightline: bool = False):
    df = pd.read_excel(path)
    n_raw = len(df)
    df.columns = [rename_column(c) for c in df.columns]

    p_cols = [c for c in df.columns if c.endswith("_p")]
    e_cols = [c for c in df.columns if c.endswith("_e")]
    item_cols = p_cols + e_cols
    demo_cols = [c for c in df.columns if c not in item_cols]

    # sap xep lai: demographics -> P theo dimension -> E theo dimension
    def sort_key(c):
        dim, idx, _ = c.rsplit("_", 2)
        return (list(DIMENSIONS.values()).index(dim), int(idx))
    df = df[demo_cols + sorted(p_cols, key=sort_key) + sorted(e_cols, key=sort_key)]
    p_cols, e_cols = sorted(p_cols, key=sort_key), sorted(e_cols, key=sort_key)
    item_cols = p_cols + e_cols

    # timestamp: so serial cua Excel -> datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="D", origin="1899-12-30",
                                         errors="coerce")

    # chuan hoa cac cot phan loai
    for col in demo_cols:
        if col != "timestamp":
            df[col] = df[col].map(lambda v, c=col: normalize_value(c, v))

    # muc hoi -> so, ngoai 1..5 coi la thieu
    df[item_cols] = df[item_cols].apply(pd.to_numeric, errors="coerce")
    df[item_cols] = df[item_cols].where(df[item_cols].isin([1, 2, 3, 4, 5]))

    df.insert(0, "respondent_id", range(1, len(df) + 1))   # id theo thu tu goc

    # ---- danh dau ly do loai (uu tien tu tren xuong) ----
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

    dropped = df[reason != ""].assign(drop_reason=reason[reason != ""])
    clean_df = df[reason == ""].reset_index(drop=True)
    clean_df[item_cols] = clean_df[item_cols].astype(int)

    # ---- bao cao ----
    print(f"Phieu goc          : {n_raw}")
    print(f"Phieu bi loai      : {len(dropped)}")
    print(dropped["drop_reason"].value_counts().to_string() if len(dropped) else "  (khong co)")
    print(f"Phieu hop le       : {len(clean_df)}")
    print(f"So cot P / E       : {len(p_cols)} / {len(e_cols)}")
    return clean_df, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", type=Path, help="file xlsx tu Google Forms")
    ap.add_argument("--drop-straightline", action="store_true",
                    help="loai ca cac phieu chon cung 1 diem (1,2,3,4) cho toan bo 32 cau")
    ap.add_argument("-o", "--outdir", type=Path, default=None, help="thu muc xuat (mac dinh: canh file goc)")
    args = ap.parse_args()

    clean_df, dropped = clean(args.input, args.drop_straightline)
    outdir = args.outdir or Path("data")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = slugify(args.input.stem)
    clean_df.to_csv(outdir / f"{stem}_clean.csv", index=False, encoding="utf-8-sig")
    dropped.to_csv(outdir / f"{stem}_dropped.csv", index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {outdir / (stem + '_clean.csv')}")
    print(f"Da luu: {outdir / (stem + '_dropped.csv')}")


if __name__ == "__main__":
    main()