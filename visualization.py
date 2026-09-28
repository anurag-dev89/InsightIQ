"""
visualization.py  —  Charts and KPI Cards
==========================================

This file renders visual components in the Streamlit UI:

    show_kpis()         → Top-level metric cards  (rows, columns, memory, missing)
    show_cleaning_kpis()→ Cleaning summary cards  (duplicates removed, nulls filled, etc.)
    show_result_kpi()   → Single-value answer card (e.g. "Total Revenue: ₹1.2M")
    show_chart()        → Auto-pick the best chart for the result (bar / line / pie / scatter)

WHY automatic charts:
    Instead of asking the user to choose a chart type, we inspect the result
    and pick the most appropriate one automatically.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards — at-a-glance numbers shown at the top of the page
# ─────────────────────────────────────────────────────────────────────────────

def show_kpis(df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    """
    Show 5 top-level KPI cards comparing original vs cleaned dataset.

    Displayed metrics: Rows · Columns · Missing Values · Memory · Numeric Columns
    Deltas show what changed during cleaning (e.g. "−2 removed").
    """
    # Calculate memory in a readable unit (KB / MB)
    mem_bytes = df.memory_usage(deep=True).sum()
    if mem_bytes < 1024 * 1024:
        mem_str = f"{mem_bytes / 1024:.1f} KB"
    else:
        mem_str = f"{mem_bytes / (1024 * 1024):.2f} MB"

    c1, c2, c3, c4, c5 = st.columns(5)

    # Show delta (difference) vs original to highlight cleaning effects
    row_diff = len(df) - len(original_df)
    row_delta = f"{row_diff:+d} after cleaning" if row_diff != 0 else None

    missing_orig = int(original_df.isna().sum().sum())
    missing_now  = int(df.isna().sum().sum())
    miss_delta   = f"{missing_now - missing_orig:+d}" if missing_orig != missing_now else "0 remaining"

    c1.metric("Rows",            f"{len(df):,}",         delta=row_delta)
    c2.metric("Columns",         f"{len(df.columns)}")
    c3.metric("Missing Values",  f"{missing_now:,}",     delta=miss_delta)
    c4.metric("Memory",          mem_str)
    c5.metric("Numeric Columns", f"{df.select_dtypes(include='number').shape[1]}")


def show_cleaning_kpis(report: dict) -> None:
    """Show KPI cards summarising what the auto-cleaning changed."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicates Removed",    report.get("duplicates_removed", 0))
    c2.metric("Empty Rows Removed",    report.get("empty_rows_removed", 0))
    c3.metric("Columns Type-Fixed",    len(report.get("columns_converted", [])))
    c4.metric("Nulls Filled",          sum(report.get("nulls_filled", {}).values()))


def show_result_kpi(result) -> bool:
    """
    If the result is a single value (scalar), show it as a big KPI card.

    Examples:
      result = 42        → shows a metric card "Answer: 42"
      result = "Phone"   → shows "Answer: Phone"
      result = DataFrame → returns False (not a scalar, render as table instead)
    """
    import numbers as num_module

    if isinstance(result, bool):
        st.metric("Answer", "Yes" if result else "No")
        return True

    if isinstance(result, num_module.Number):
        # Format nicely: large numbers with commas, small floats with 4 sig figs
        formatted = f"{result:,.4g}" if isinstance(result, float) else f"{result:,}"
        st.metric("Answer", formatted)
        return True

    if isinstance(result, str) and len(result) < 80:
        st.metric("Answer", result)
        return True

    # Single-cell DataFrame → treat it as a scalar
    if isinstance(result, pd.DataFrame) and result.shape == (1, 1):
        val = result.iat[0, 0]
        st.metric(str(result.columns[0]), f"{val:,.4g}" if isinstance(val, float) else str(val))
        return True

    # Single-element Series → also a scalar
    if isinstance(result, pd.Series) and len(result) == 1:
        val = result.iloc[0]
        st.metric(str(result.name or "Result"), str(val))
        return True

    return False  # Not a scalar — caller should render as a table


# ─────────────────────────────────────────────────────────────────────────────
# Automatic Chart Selection
# ─────────────────────────────────────────────────────────────────────────────

def show_chart(result) -> bool:
    """
    Automatically pick and render the best chart for the given result.

    Decision logic:
      Series (numeric)          → Bar chart  (values on y-axis, index on x-axis)
      Series (text/categorical) → Bar chart  (frequency / value counts)
      DataFrame + datetime col  → Line chart (trend over time)
      DataFrame + category col  → Bar chart  (grouped by category)
      DataFrame + 2 numeric cols→ Scatter chart (correlation)
      Anything else             → Skip chart

    Returns True if a chart was rendered, False if not.
    """
    try:
        import plotly.express as px
    except ImportError:
        # Plotly not installed — fall back to Streamlit's built-in charts
        return _fallback_chart(result)

    # ── Case 1: pandas Series ─────────────────────────────────────────────────
    if isinstance(result, pd.Series):
        series = result.dropna().head(30)   # Show top 30 for readability
        if series.empty:
            return False

        name = str(series.name or "Value")
        idx  = str(series.index.name or "Category")
        plot_df = series.reset_index()
        plot_df.columns = [idx, name]

        if pd.api.types.is_numeric_dtype(series):
            fig = px.bar(plot_df, x=idx, y=name, title=f"{name} by {idx}", color=name)
        else:
            # Count occurrences of each category
            counts = series.astype(str).value_counts().head(20).reset_index()
            counts.columns = [name, "Count"]
            fig = px.bar(counts, x=name, y="Count", title=f"Distribution of {name}")

        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        return True

    # ── Case 2: pandas DataFrame ──────────────────────────────────────────────
    if not isinstance(result, pd.DataFrame) or result.empty or len(result) < 2:
        return False

    df = result.head(100)  # Cap at 100 rows to keep charts readable
    numeric_cols    = df.select_dtypes(include="number").columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]

    if not numeric_cols:
        return False  # Nothing to chart without numbers

    # Case 2a: Has a datetime column → Line chart (time series)
    if datetime_cols:
        date_col = datetime_cols[0]
        y_cols   = numeric_cols[:3]   # Plot up to 3 numeric lines
        fig = px.line(
            df.sort_values(date_col),
            x=date_col, y=y_cols,
            title=f"{', '.join(y_cols)} over time",
            markers=True,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        return True

    # Case 2b: Has a category column → Bar chart grouped by category
    if categorical_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        n_cats  = df[cat_col].nunique()

        # If very few categories (≤6), use a pie / donut chart
        if n_cats <= 6 and len(numeric_cols) == 1 and (df[num_col] >= 0).all():
            agg = df.groupby(cat_col)[num_col].sum().reset_index()
            fig = px.pie(agg, names=cat_col, values=num_col,
                         title=f"{num_col} share by {cat_col}", hole=0.4)
        else:
            agg = df.groupby(cat_col)[numeric_cols[:2]].sum().reset_index()
            agg = agg.sort_values(num_col, ascending=False).head(20)
            fig = px.bar(agg, x=cat_col, y=numeric_cols[:2],
                         barmode="group", title=f"{num_col} by {cat_col}")

        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        return True

    # Case 2c: Two numeric columns → Scatter plot (correlation)
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        return True

    return False


def _fallback_chart(result) -> bool:
    """Simple fallback using Streamlit's built-in chart functions (no Plotly needed)."""
    if isinstance(result, pd.Series) and pd.api.types.is_numeric_dtype(result):
        st.bar_chart(result.dropna().head(30))
        return True
    if isinstance(result, pd.DataFrame):
        numeric_cols = result.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.line_chart(result[numeric_cols].head(100))
            return True
    return False
