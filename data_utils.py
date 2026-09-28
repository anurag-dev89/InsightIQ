"""
data_utils.py  —  Data Loading, Cleaning, and Export
======================================================

This file handles everything to do with data before the AI sees it:

    load_csv()    → Read a CSV file safely (enforces size limits)
    clean_data()  → Auto-clean: remove duplicates, fix types, fill missing values
    get_stats()   → Get descriptive statistics for numeric columns
    export_pdf()  → Generate a downloadable PDF of the analysis result

WHY auto-cleaning matters:
    Real-world data is messy. Dates stored as strings, numbers with spaces,
    duplicate rows — all of this confuses the AI. We clean first, then analyse.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


# ── Upload limits (we don't want huge files crashing the app) ─────────────────

MAX_FILE_SIZE_MB = 50
MAX_ROWS        = 250_000
MAX_COLUMNS     = 200


# ─────────────────────────────────────────────────────────────────────────────
# Load CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Supports:
      - A file path (str or Path)
      - A file-like object (from Streamlit's file uploader)

    Enforces size limits so large files don't crash the app.
    """
    # If it's a file path, check it exists and isn't too large
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File is {size_mb:.1f} MB — limit is {MAX_FILE_SIZE_MB} MB.")
    else:
        # Streamlit UploadedFile has a .size attribute (in bytes)
        size_mb = getattr(source, "size", 0) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File is {size_mb:.1f} MB — limit is {MAX_FILE_SIZE_MB} MB.")
        # Rewind to the start so pandas can read it
        if hasattr(source, "seek"):
            source.seek(0)

    # Try UTF-8 first, fall back to Latin-1 for files with special characters
    try:
        df = pd.read_csv(source, nrows=MAX_ROWS, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, nrows=MAX_ROWS, encoding="latin-1", low_memory=False)

    # Basic checks on the loaded data
    if df.empty:
        raise ValueError("The CSV file is empty or has no data rows.")
    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"File has {len(df.columns)} columns — limit is {MAX_COLUMNS}.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Clean Data
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Automatically clean the dataset and return a cleaning report.

    Cleaning steps (in order):
      1. Remove completely empty rows (all values are NaN)
      2. Remove duplicate rows (exact copies)
      3. Trim whitespace from text columns  ("  Apple  " → "Apple")
      4. Convert text columns that look like numbers to numeric type
      5. Convert text columns that look like dates to datetime type
      6. Fill missing numbers with the column median
      7. Fill missing text values with the column mode (most common value)

    Returns:
      cleaned_df   — the fixed-up DataFrame ready for analysis
      report       — a dictionary summarising what was changed
    """
    cleaned = df.copy()  # Always work on a copy — never modify the original

    # We'll track what changes we make for the cleaning report
    report = {
        "original_rows":       len(df),
        "empty_rows_removed":  0,
        "duplicates_removed":  0,
        "columns_converted":   [],   # columns whose dtype was changed
        "nulls_filled":        {},   # column → how many nulls were filled
    }

    # ── Step 1: Remove completely empty rows ─────────────────────────────────
    # A row where every single cell is NaN is useless — remove it
    empty_mask = cleaned.isna().all(axis=1)
    report["empty_rows_removed"] = int(empty_mask.sum())
    cleaned = cleaned[~empty_mask].reset_index(drop=True)

    # ── Step 2: Remove duplicate rows ────────────────────────────────────────
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(cleaned)

    # ── Step 3: Trim whitespace from text columns ─────────────────────────────
    # "  Apple  " becomes "Apple" — prevents mismatched groupby results
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].str.strip()

    # ── Step 4: Convert text → numeric where possible ─────────────────────────
    # If a column holds "123", "456", it should be int/float not string
    for col in cleaned.select_dtypes(include="object").columns:
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        # Only convert if ALL non-null values successfully became numbers
        non_null = cleaned[col].dropna()
        if len(non_null) > 0 and converted.notna().sum() == len(non_null):
            cleaned[col] = converted
            report["columns_converted"].append(f"{col} → numeric")

    # ── Step 5: Convert text → datetime where possible ───────────────────────
    # Detect date columns by name ("date", "time", "created_at") or content
    date_keywords = {"date", "time", "timestamp", "day", "month", "year", "dob"}
    for col in cleaned.select_dtypes(include="object").columns:
        col_lower = col.lower()
        has_date_keyword = any(kw in col_lower for kw in date_keywords)

        if has_date_keyword:
            try:
                converted = pd.to_datetime(cleaned[col], errors="coerce")
                non_null = cleaned[col].dropna()
                # Only convert if all values parsed successfully as dates
                if len(non_null) > 0 and converted.notna().sum() == len(non_null):
                    cleaned[col] = converted
                    report["columns_converted"].append(f"{col} → datetime")
            except Exception:
                pass  # If conversion fails, leave the column as-is

    # ── Step 6: Fill missing numbers with the column median ───────────────────
    # Median is better than mean for skewed data (less affected by outliers)
    for col in cleaned.select_dtypes(include="number").columns:
        null_count = cleaned[col].isna().sum()
        if null_count > 0:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            report["nulls_filled"][col] = int(null_count)

    # ── Step 7: Fill missing text with the most common value (mode) ───────────
    for col in cleaned.select_dtypes(include="object").columns:
        null_count = cleaned[col].isna().sum()
        if null_count > 0:
            mode_val = cleaned[col].mode(dropna=True)
            if not mode_val.empty:
                cleaned[col] = cleaned[col].fillna(mode_val.iloc[0])
                report["nulls_filled"][col] = int(null_count)

    # Final summary numbers for the report
    report["cleaned_rows"]     = len(cleaned)
    report["total_nulls_before"] = int(df.isna().sum().sum())
    report["total_nulls_after"]  = int(cleaned.isna().sum().sum())

    return cleaned, report


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for all numeric columns.

    This calls pandas' built-in .describe() and renames the columns
    to be more human-readable (e.g. '50%' → 'Median').
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()  # No numeric columns to summarise

    stats = numeric_df.describe().T   # .T transposes so columns are the stat names
    stats = stats.rename(columns={
        "count": "Count",
        "mean":  "Mean",
        "std":   "Std Dev",
        "min":   "Min",
        "25%":   "25th %",
        "50%":   "Median",
        "75%":   "75th %",
        "max":   "Max",
    })
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PDF Export
# ─────────────────────────────────────────────────────────────────────────────

def export_pdf(question: str, code: str, result) -> bytes:
    """
    Generate a PDF report containing the question, result table, and code.

    Uses ReportLab to build the PDF entirely in memory (no temp files).
    Returns the PDF as bytes so Streamlit can offer it as a download.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            LongTable, Paragraph, Preformatted,
            SimpleDocTemplate, Spacer, TableStyle,
        )
    except ImportError:
        raise RuntimeError("ReportLab not installed. Run: pip install -r requirements.txt")

    # Helper: make a value safe to put in a PDF (strip control characters)
    def _safe(val) -> str:
        from xml.sax.saxutils import escape
        text = "".join(c if c in "\n\r\t" or ord(c) >= 32 else " " for c in str(val))
        return escape(text.encode("cp1252", "replace").decode("cp1252"))

    # Convert the result to a DataFrame so we can display it as a table
    if isinstance(result, pd.DataFrame):
        frame = result.head(100)
    elif isinstance(result, pd.Series):
        frame = result.head(100).to_frame(name=result.name or "Result")
    elif isinstance(result, dict):
        frame = pd.DataFrame(list(result.items())[:100], columns=["Key", "Value"])
    else:
        frame = pd.DataFrame({"Answer": [result]})

    # Build the PDF
    output = BytesIO()
    doc    = SimpleDocTemplate(
        output,
        pagesize=landscape(letter),
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        title="AI Data Analyst Report",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Cell",      parent=styles["BodyText"], fontSize=7, leading=9))
    styles.add(ParagraphStyle("CodeSmall", parent=styles["Code"],     fontSize=7, leading=9, alignment=TA_LEFT))

    # Build the result table
    col_widths = [doc.width / max(1, len(frame.columns))] * len(frame.columns)
    table_data = [[Paragraph(_safe(c), styles["Cell"]) for c in frame.columns]]
    for row in frame.itertuples(index=False, name=None):
        table_data.append([Paragraph(_safe(str(v)[:500]), styles["Cell"]) for v in row])

    table = LongTable(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#b7c9d6")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f9")]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))

    story = [
        Paragraph("AI Data Analyst Report", styles["Title"]),
        Spacer(1, 0.1 * inch),
        Paragraph(f"<b>Question:</b> {_safe(question)}", styles["BodyText"]),
        Spacer(1, 0.1 * inch),
        Paragraph("<b>Result:</b>", styles["Heading2"]),
        table,
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Generated Code:</b>", styles["Heading2"]),
        Preformatted(_safe(code[:6000]), styles["CodeSmall"], maxLineLength=110),
    ]

    doc.build(story)
    return output.getvalue()
