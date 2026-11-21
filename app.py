"""FinSight: a local-first finance and banking clustering dashboard."""
from __future__ import annotations

import os
import hashlib
import html
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.spark_engine import cluster_dataset, create_spark, inspect_dataset, read_dataset

px.defaults.color_discrete_sequence = ["#147D92", "#38A887", "#F0AE47", "#6678CE", "#D47769", "#7960A8"]

st.set_page_config(page_title="FinSight | Customer Segmentation", page_icon="◈", layout="wide")
with st.sidebar:
    st.markdown("## ◈ FinSight")
    st.caption("LOCAL ANALYTICS WORKSPACE")
    st.session_state.setdefault("dark_mode", True)
    dark_mode = st.toggle("Dark mode", key="dark_mode")
    st.divider()
    uploaded = st.file_uploader("Upload a dataset", type=["csv", "parquet"], help="Use a tabular finance or banking dataset. The app inspects the uploaded schema; it does not assume fixed column names.")
    st.caption("CSV and Parquet · Processed locally with Spark")
    st.markdown("<div class='nav-heading'>WORKSPACE</div>", unsafe_allow_html=True)
    workspace_page = st.radio(
        "Workspace section",
        ["Explore data", "Financial summaries", "Trends over time", "Clustering", "Spark SQL"],
        format_func=lambda page: {"Explore data": "▦   Explore data", "Financial summaries": "＄   Financial summaries", "Trends over time": "◷   Trends over time", "Clustering": "✳   Clustering", "Spark SQL": "⌘   Spark SQL"}[page],
        key="workspace_page",
        label_visibility="collapsed",
    )
    if workspace_page == "Clustering":
        st.markdown("<div class='nav-subheading'>CLUSTERING</div>", unsafe_allow_html=True)
        clustering_page = st.radio(
            "Clustering page",
            ["Run K-Means", "Compare K values"],
            format_func=lambda page: {"Run K-Means": "◎   Run K-Means", "Compare K values": "↗   Compare K values"}[page],
            key="clustering_page_choice",
            label_visibility="collapsed",
        )
    else:
        clustering_page = None

if dark_mode:
    page, panel, text, muted, border, field, scheme = "#0b1422", "#111f31", "#e8eff7", "#afbdcc", "#2b4058", "#14253a", "dark"
    px.defaults.template = "plotly_dark"
    canvas_bg = "radial-gradient(ellipse 48% 42% at 0% 0%,#0a4860 0%,transparent 72%),radial-gradient(ellipse 42% 55% at 100% 30%,#30205c 0%,transparent 72%),#080d19"
else:
    page, panel, text, muted, border, field, scheme = "#f3f7fc", "#ffffff", "#152a43", "#52677d", "#d8e2ed", "#ffffff", "light"
    px.defaults.template = "plotly_white"
    canvas_bg = "linear-gradient(180deg,#f6f9fd 0%,#edf3fa 100%)"

st.markdown(f"""
<style>
:root {{ color-scheme:{scheme}; }}
[data-testid="stAppViewContainer"] {{background:{canvas_bg};color:{text};font-family:Inter,'Segoe UI',sans-serif}}
.stApp {{background:transparent;color:{text};font-family:Inter,'Segoe UI',sans-serif}}
[data-testid="stHeader"] {{background:transparent}}
main .block-container {{max-width:1640px;padding-top:.8rem;padding-bottom:3rem}}
section[data-testid="stSidebar"] {{min-width:270px!important;max-width:290px!important;background:linear-gradient(180deg,#09182a 0%,#0d2137 58%,#102b42 100%)}}
[data-testid="stSidebar"] * {{color:#edf5fc}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{color:#edf5fc!important}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:#c0d0df!important;opacity:1}}
.nav-heading {{margin:1rem 0 .55rem;color:#86a2ba;font-size:.68rem;font-weight:800;letter-spacing:.16em}}
.nav-subheading {{margin:.9rem 0 .4rem .65rem;padding-left:.65rem;border-left:2px solid #31b9cf;color:#7fd7da;font-size:.65rem;font-weight:800;letter-spacing:.14em}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{gap:.3rem}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{padding:.55rem .65rem;border:1px solid transparent;border-radius:11px;transition:all .16s ease}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{background:#ffffff0d;border-color:#ffffff18}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{background:linear-gradient(100deg,#16889938,#8b5cf620);border-color:#4cbac544}}
[data-testid="stFileUploaderDropzone"] {{background:#10243a!important;border:1px dashed #45647d!important;border-radius:15px!important;color:#e9f1f8!important}}
[data-testid="stFileUploaderDropzone"] * {{color:#dce8f2!important;opacity:1!important}}
[data-testid="stFileUploaderFile"] {{background:#18344d!important;border:1px solid #385a73!important;border-radius:11px!important}}
[data-testid="stFileUploaderFile"] * {{color:#f0f6fb!important;opacity:1!important}}
[data-testid="stFileUploaderDropzone"] button {{background:linear-gradient(110deg,#31b9cf,#37c6ae)!important;color:#082032!important;border:0!important;border-radius:9px!important}}
[data-testid="stFileUploaderDropzone"] button * {{color:#082032!important}}
h1,h2,h3,h4 {{color:{text};font-family:Inter,'Segoe UI',sans-serif;letter-spacing:-.025em}}
.hero {{position:relative;overflow:hidden;padding:1.05rem 1.55rem;border:1px solid #31526b;border-radius:18px;background:radial-gradient(ellipse at 90% 0%,#247d83 0%,transparent 35%),linear-gradient(115deg,#10243d 0%,#173b58 62%,#205c69 100%);color:#fff;margin:.15rem 0 1rem;box-shadow:0 14px 34px #13334d30}}
.hero:after {{content:'◈';position:absolute;right:4%;top:-42px;font-size:170px;line-height:1;color:#ffffff12}}
.hero-kicker {{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#9de1d4;font-weight:700;margin-bottom:.55rem}}
.hero h1 {{color:#fff;margin:0;font-size:1.8rem;font-weight:750;letter-spacing:-.04em}}.hero p {{color:#d2e1ec;margin:.3rem 0 0;font-size:.92rem}}
.welcome {{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:1.7rem 2rem;margin:0 0 1.3rem;border-radius:18px;background:linear-gradient(115deg,#173b58,#205c69);color:#fff;box-shadow:0 12px 30px #13334d1c}}
.welcome-kicker,.section-kicker {{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#36b3a6;font-weight:750}}
.welcome h2 {{margin:.35rem 0;color:#fff;font-size:1.55rem;letter-spacing:-.03em}}
.welcome p {{margin:0;color:#d6e4ee;max-width:760px}}
.welcome-mark {{font-size:5rem;color:#b2ede533;padding-right:.3rem}}
.section-title {{margin:.15rem 0 .9rem;font-size:1.25rem;font-weight:700}}
.feature-card,.step-card {{height:100%;min-height:138px;padding:1.05rem 1.1rem;border-radius:15px;background:{panel};border:1px solid {border};box-shadow:0 7px 20px #18395a0a}}
.feature-icon {{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;margin-bottom:.65rem;border-radius:10px;background:#1688991a;color:#168899;font-weight:800;font-size:.8rem}}
.feature-card h3,.step-card h3 {{margin:0 0 .35rem;color:{text};font-size:1rem;letter-spacing:-.01em}}
.feature-card p,.step-card p {{margin:0;color:{muted};font-size:.88rem;line-height:1.5}}
.step-card {{min-height:125px}}
.step-index {{color:#168899;font-weight:800;font-size:.75rem;letter-spacing:.1em;margin-bottom:.55rem}}
[data-baseweb="tab-list"] {{background:{panel};padding:.35rem;border:1px solid {border};border-radius:14px;gap:.25rem}}
[data-baseweb="tab"] {{border-radius:10px;color:{muted};padding:.55rem 1rem}}
[data-baseweb="tab"][aria-selected="true"] {{background:{field};color:#45c2d2;border:1px solid {border}}}
[role="dialog"],div[aria-modal="true"] {{background:{panel};color:{text};border:1px solid {border};border-radius:18px}}
[role="dialog"] h1,[role="dialog"] h2,[role="dialog"] p {{color:{text}}}
[data-testid="stMetric"] {{background:{panel}!important;border:1px solid {border}!important;padding:1rem 1.15rem;border-radius:16px;box-shadow:0 5px 18px #18395a12}}
[data-testid="stMetricLabel"] p {{color:{muted}!important;font-weight:600}}
[data-testid="stMetricValue"] {{color:{text}!important;font-weight:700}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{background:{panel}!important;border:1px solid {border}!important;border-radius:14px;overflow:hidden;color-scheme:{scheme}}}
[data-testid="stDataFrame"] * {{color-scheme:{scheme}}}
.finsight-table-wrap {{max-height:440px;overflow:auto;border:1px solid {border};border-radius:14px;background:{panel};box-shadow:0 8px 24px #07152518}}
.finsight-table {{width:100%;border-collapse:separate;border-spacing:0;background:{panel};color:{text};font-size:.9rem}}
.finsight-table th {{position:sticky;top:0;z-index:1;padding:.8rem .9rem;background:{field};color:{text};text-align:left;font-weight:700;border-bottom:1px solid {border};white-space:nowrap}}
.finsight-table td {{padding:.68rem .9rem;color:{text};border-bottom:1px solid {border};white-space:nowrap}}
.finsight-table tbody tr:nth-child(even) td {{background:{field}}}
.finsight-table tbody tr:hover td {{background:#16889920}}
[data-testid="stWidgetLabel"] p {{color:{text}!important;font-weight:600}}
[data-testid="stCaptionContainer"] p {{color:{muted}!important}}
[data-baseweb="tab-list"] {{gap:.35rem;border-bottom:1px solid {border}}}
[data-baseweb="tab"] {{color:{muted}}}
[data-baseweb="tab"][aria-selected="true"] {{color:#31a5a5;border-bottom-color:#31a5a5;font-weight:700}}
[data-baseweb="select"] > div, [data-testid="stMultiSelect"] [role="combobox"] {{background:{field};color:{text};border-color:{border}}}
[data-baseweb="popover"], [data-baseweb="popover"] [role="option"] {{background:{panel};color:{text}}}
[data-testid="stTextArea"] textarea {{background:{field}!important;color:{text}!important;border:1px solid {border}!important;border-radius:12px!important}}
[data-testid="stSlider"] [role="slider"] {{border-color:#31a5a5}}
div.stButton>button {{border-radius:11px;font-weight:650;border:1px solid {border};transition:all .16s ease}}
div.stButton>button[kind="primary"] {{background:linear-gradient(110deg,#176d82,#218f91);border:0;color:#fff;box-shadow:0 5px 14px #176d8228}}
div.stButton>button:hover {{border-color:#31a5a5;color:#31a5a5;transform:translateY(-1px)}}
[data-testid="stDownloadButton"] button {{border-radius:10px}}
[data-testid="stCode"] {{border-radius:12px}}
div[data-testid="stAlert"] {{border-radius:12px;background:{panel};color:{text};border:1px solid {border}}}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] svg {{color:{text}!important}}
</style>
""", unsafe_allow_html=True)

def display_table(frame: pd.DataFrame, max_height: int = 440) -> None:
    """Render a compact, escaped table using the active app theme."""
    markup = frame.to_html(index=False, escape=True, classes="finsight-table", border=0)
    st.markdown(
        f'<div class="finsight-table-wrap" style="max-height:{max_height}px">{markup}</div>',
        unsafe_allow_html=True,
    )

def build_analysis_report(upload_name: str, upload_size: int, page_name: str, info: dict, schema: pd.DataFrame, sections: list[tuple[str, object]]) -> str:
    """Build a self-contained, printable HTML report from aggregate results."""
    overview = pd.DataFrame([{
        "Uploaded dataset": upload_name,
        "File size (MB)": round(upload_size / (1024 * 1024), 2),
        "Records": info["rows"],
        "Columns": info["columns"],
        "Duplicate rows": info["duplicates"],
        "Missing cells": sum(info["missing"].values()),
        "Dashboard page": page_name,
    }])
    blocks = []
    plotly_included = False
    all_sections = [("Dataset overview", overview), ("Detected schema and missing values", schema), *sections]
    if not sections:
        all_sections.append(("Page analysis", "No page-specific aggregate result is available yet. This report includes the dataset overview and detected schema."))
    for heading, content in all_sections:
        if isinstance(content, pd.DataFrame):
            body = content.to_html(index=False, escape=True, border=0, classes="report-table", na_rep="—")
        elif isinstance(content, go.Figure):
            body = content.to_html(
                full_html=False,
                include_plotlyjs=not plotly_included,
                config={"responsive": True, "displaylogo": False},
            )
            plotly_included = True
        else:
            body = f"<p>{html.escape(str(content))}</p>"
        blocks.append(f"<section><h2>{html.escape(heading)}</h2>{body}</section>")
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinSight Analysis Report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f8;color:#16283d;font:15px/1.55 'Segoe UI',Arial,sans-serif}}
.page{{max-width:1080px;margin:36px auto;padding:0 22px}}header{{padding:30px 34px;border-radius:18px;background:linear-gradient(115deg,#10243d,#205c69);color:#fff;box-shadow:0 12px 30px #13334d22}}
.kicker{{color:#9de1d4;font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase}}h1{{margin:8px 0 0;font-size:32px}}header p{{margin:6px 0 0;color:#d2e1ec}}.stamp{{margin-top:12px;color:#b9ccda;font-size:13px}}
section{{margin:20px 0;padding:22px;background:#fff;border:1px solid #d8e2ed;border-radius:15px;box-shadow:0 6px 18px #18395a0b;overflow-x:auto}}h2{{margin:0 0 14px;font-size:19px;color:#16344e}}
.report-table{{width:100%;border-collapse:collapse;font-size:13px}}.report-table th{{background:#eaf2f8;color:#1a4059;text-align:left;font-weight:700}}.report-table th,.report-table td{{padding:10px 12px;border-bottom:1px solid #e1e8ef;vertical-align:top}}.report-table tr:last-child td{{border-bottom:0}}
footer{{padding:12px 4px 30px;color:#60758a;font-size:12px}}@media print{{body{{background:#fff}}.page{{max-width:none;margin:0;padding:0}}header,section{{box-shadow:none;break-inside:avoid}}}}
</style></head><body><main class="page"><header><div class="kicker">FinSight · Finance analytics powered by Spark</div><h1>Analysis report</h1><p>{html.escape(upload_name)}</p><div class="stamp">Generated {html.escape(generated_at)}</div></header>{''.join(blocks)}<footer>Contains dataset metadata and aggregate analysis only; individual uploaded records are not included. Interpret totals and patterns using the meaning and level of detail of your source data. To create a PDF, open this report in a browser and choose Print → Save as PDF.</footer></main></body></html>"""

def select_recommended_k(k_value: int) -> None:
    st.session_state["selected_k"] = int(k_value)
    st.session_state["clustering_page_choice"] = "Run K-Means"

st.markdown('<div class="hero"><div class="hero-kicker">FINANCE ANALYTICS · POWERED BY SPARK</div><h1>FinSight</h1><p>See the patterns behind your financial data.</p></div>', unsafe_allow_html=True)

if not uploaded:
    st.session_state.pop("cluster_result", None)
    st.session_state.pop("active_upload", None)
    st.session_state.pop("confirmed_upload", None)
    st.session_state.pop("cancelled_upload", None)
    st.session_state.pop("confirm_dataset", None)
    previous_path = st.session_state.pop("upload_temp_path", None)
    if previous_path and os.path.exists(previous_path):
        os.unlink(previous_path)
    st.markdown("""
    <div class="welcome">
      <div><div class="welcome-kicker">YOUR LOCAL ANALYTICS STUDIO</div>
      <h2>Bring your financial data into focus.</h2>
      <p>Upload a CSV or Parquet file in the sidebar. Review it first; Spark only starts analyzing after you confirm.</p></div>
      <div class="welcome-mark">◈</div>
    </div>
    <div class="section-kicker">BUILT FOR YOUR WORKFLOW</div>
    """, unsafe_allow_html=True)
    a,b,c = st.columns(3, gap="medium")
    with a:
        st.markdown('<div class="feature-card"><div class="feature-icon">DATA</div><h3>Bring your own data</h3><p>Explore CSV or Parquet files with whatever columns your dataset contains.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="feature-card"><div class="feature-icon">SQL</div><h3>Ask your data</h3><p>Run Spark SQL to filter, group, count, and profile your records.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="feature-card"><div class="feature-icon">K</div><h3>Find useful groups</h3><p>Compare K values, profile segments, and export the results.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker" style="margin-top:1.5rem">FOUR SIMPLE STEPS</div>', unsafe_allow_html=True)
    steps = st.columns(4, gap="small")
    step_data = [("01", "Upload", "Choose a CSV or Parquet dataset."), ("02", "Review", "Confirm before analysis begins."), ("03", "Explore", "Inspect quality and query with SQL."), ("04", "Segment", "Run K-Means and export clusters.")]
    for column, (number, title, description) in zip(steps, step_data):
        with column:
            st.markdown(f'<div class="step-card"><div class="step-index">{number} / 04</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.stop()

uploaded_bytes = uploaded.getvalue()
upload_identity = (uploaded.name, uploaded.size, hashlib.sha256(uploaded_bytes).hexdigest())
if st.session_state.get("active_upload") != upload_identity:
    st.session_state.pop("cluster_result", None)
    st.session_state.pop("cluster_score", None)
    st.session_state.pop("confirmed_upload", None)
    st.session_state.pop("cancelled_upload", None)
    st.session_state.pop("k_comparison", None)
    st.session_state.pop("k_comparison_key", None)
    st.session_state.pop("compare_max_k", None)
    st.session_state.pop("selected_k", None)
    st.session_state.pop("cluster_features_input", None)
    st.session_state.pop("outcome_column", None)
    st.session_state.pop("financial_measure", None)
    st.session_state.pop("financial_group_by", None)
    st.session_state.pop("financial_top_groups", None)
    st.session_state.pop("trend_date_column", None)
    st.session_state.pop("trend_date_format", None)
    st.session_state.pop("trend_custom_date_format", None)
    st.session_state.pop("trend_date_mode", None)
    st.session_state.pop("trend_month_column", None)
    st.session_state.pop("trend_year_column", None)
    st.session_state.pop("trend_day_column", None)
    st.session_state.pop("trend_measure", None)
    st.session_state.pop("trend_aggregation", None)
    st.session_state.pop("trend_group_by", None)
    st.session_state.pop("trend_granularity", None)
    st.session_state.pop("segment_drilldown_cluster", None)
    st.session_state.pop("segment_category_feature", None)
    st.session_state.pop("segment_numeric_feature", None)
    previous_path = st.session_state.pop("upload_temp_path", None)
    if previous_path and os.path.exists(previous_path):
        os.unlink(previous_path)
    st.session_state["active_upload"] = upload_identity
    suffix = ".parquet" if uploaded.name.lower().endswith(".parquet") else ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_bytes)
        st.session_state["upload_temp_path"] = temp_file.name

if st.session_state.get("confirmed_upload") != upload_identity:
    if st.session_state.get("cancelled_upload") == upload_identity:
        st.warning("Analysis paused. Your file has not been inspected.")
        if st.button("Review this upload", type="primary"):
            st.session_state.pop("cancelled_upload", None)
            st.rerun()
    else:
        @st.dialog("Analyze this dataset?", width="small")
        def confirm_upload_dialog():
            st.markdown(f"**{uploaded.name}**")
            st.caption(f"{uploaded.size / (1024 * 1024):.2f} MB · CSV or Parquet")
            st.write("Spark will inspect and analyze this file on your computer.")
            cancel, proceed = st.columns(2)
            with cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["cancelled_upload"] = upload_identity
                    st.rerun()
            with proceed:
                if st.button("Analyze file", type="primary", use_container_width=True):
                    st.session_state["confirmed_upload"] = upload_identity
                    st.session_state.pop("cancelled_upload", None)
                    st.rerun()
        confirm_upload_dialog()
    st.stop()

temp_path = st.session_state["upload_temp_path"]
try:
    if "spark" not in st.session_state:
        st.session_state.spark = create_spark()
    spark = st.session_state.spark
    df = read_dataset(spark, temp_path)
    info = inspect_dataset(df)
except Exception as exc:
    st.error(f"Could not read this dataset: {exc}")
    st.stop()

st.subheader("Dataset overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Records", f"{info['rows']:,}")
c2.metric("Columns", info["columns"])
c3.metric("Duplicate rows", f"{info['duplicates']:,}")
c4.metric("Missing cells", f"{sum(info['missing'].values()):,}")

schema_report = pd.DataFrame([
    {"Column": field.name, "Detected type": field.dataType.simpleString(), "Missing": info["missing"].get(field.name, 0)}
    for field in df.schema.fields
])
report_sections: list[tuple[str, object]] = []

if workspace_page == "Explore data":
    st.markdown("#### Schema detected from your upload")
    schema_pd = schema_report.copy()
    schema_pd["Suggested role"] = ["Numeric" if name in info["numeric"] else "Categorical / identifier" for name in schema_pd["Column"]]
    display_table(schema_pd)
    st.markdown("#### Data preview")
    display_table(df.limit(100).toPandas())
    st.caption("No records are silently removed. Numeric missing values are imputed during modeling; categorical missing values are handled by Spark's invalid-category option.")

elif workspace_page == "Financial summaries":
    st.markdown("#### Financial summaries")
    st.caption("Choose a numeric measure from your uploaded data. These are descriptive calculations; check what each row and field mean before interpreting totals.")
    if not info["numeric"]:
        st.info("This dataset has no detected numeric columns to summarize.")
    else:
        measure = st.selectbox("Numeric measure", options=info["numeric"], key="financial_measure")
        group_options = ["None"] + list(info.get("categorical", []))
        group_by = st.selectbox(
            "Break down by (optional)",
            options=group_options,
            key="financial_group_by",
            help="Pick a category such as a product, region, or status if your dataset has one.",
        )
        quoted_measure = "`" + measure.replace("`", "``") + "`"
        measure_col = F.col(quoted_measure)
        measure_summary = df.agg(
            F.count(measure_col).alias("Non-missing records"),
            F.round(F.sum(measure_col), 2).alias(f"Sum · {measure}"),
            F.round(F.avg(measure_col), 2).alias(f"Average · {measure}"),
            F.round(F.expr(f"percentile_approx({quoted_measure}, 0.5, 10000)"), 2).alias(f"Median · {measure}"),
            F.round(F.min(measure_col), 2).alias(f"Minimum · {measure}"),
            F.round(F.max(measure_col), 2).alias(f"Maximum · {measure}"),
        ).toPandas()
        display_table(measure_summary)
        report_sections.append(("Financial measure summary", measure_summary))
        st.caption("The sum is an arithmetic total across rows. It may not represent consolidated financial value, especially when rows are balances, snapshots, or repeated account records.")

        if group_by != "None":
            st.markdown(f"#### {measure} by {group_by}")
            max_groups = st.slider("Categories to show", min_value=5, max_value=25, value=10, key="financial_top_groups")
            quoted_group = "`" + group_by.replace("`", "``") + "`"
            group_col = F.coalesce(F.col(quoted_group).cast("string"), F.lit("Missing"))
            grouped = (
                df.groupBy(group_col.alias("Category"))
                .agg(
                    F.count(F.lit(1)).alias("Records"),
                    F.count(measure_col).alias("Non-missing measures"),
                    F.round(F.sum(measure_col), 2).alias(f"Sum · {measure}"),
                    F.round(F.avg(measure_col), 2).alias(f"Average · {measure}"),
                )
                .orderBy(F.col(f"Sum · {measure}").desc_nulls_last())
                .limit(max_groups)
            )
            grouped_pd = grouped.toPandas()
            display_table(grouped_pd)
            report_sections.append((f"{measure} by {group_by}", grouped_pd))
            if not grouped_pd.empty and grouped_pd[f"Sum · {measure}"].notna().any():
                financial_fig = px.bar(
                    grouped_pd,
                    x="Category",
                    y=f"Sum · {measure}",
                    title=f"Sum of {measure} by {group_by} (top {len(grouped_pd)})",
                    color="Category",
                    color_discrete_sequence=["#31b9cf", "#9a6cff", "#42c6a5", "#efb750", "#ed7b8e"],
                )
                financial_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color": text, "family": "Inter, Segoe UI, sans-serif"}, title_font={"color": text, "size": 16}, showlegend=False, margin={"l": 16, "r": 12, "t": 48, "b": 16})
                financial_fig.update_xaxes(showgrid=False, linecolor=border, tickfont_color=muted, tickangle=-25)
                financial_fig.update_yaxes(gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_font_color=muted)
                st.plotly_chart(financial_fig, use_container_width=True)
                report_sections.append((f"Sum of {measure} by {group_by} chart", financial_fig))

elif workspace_page == "Trends over time":
    st.markdown("#### Trends over time")
    st.caption("Choose a date or time field and a measure to see how your data changes. The app uses your uploaded schema and column values.")
    temporal_candidates = [
        field.name
        for field in df.schema.fields
        if field.dataType.simpleString() == "string"
        or field.dataType.simpleString() == "date"
        or field.dataType.simpleString().startswith("timestamp")
    ]
    date_mode = st.radio("How is the date stored?", ["One date/time column", "Separate date parts"], horizontal=True, key="trend_date_mode")
    if date_mode == "One date/time column" and not temporal_candidates:
        st.info("No text or date/time columns were detected. If your date is split across fields, choose Separate date parts.")
    else:
        date_part_columns = set()
        if date_mode == "Separate date parts":
            month_guess = next((column for column in df.columns if "month" in column.lower()), df.columns[0])
            month_column = st.selectbox("Month column", options=df.columns, index=list(df.columns).index(month_guess), key="trend_month_column")
            remaining_columns = [column for column in df.columns if column != month_column]
            year_options = ["None"] + remaining_columns
            if st.session_state.get("trend_year_column") not in year_options:
                st.session_state.pop("trend_year_column", None)
            year_guess = next((column for column in remaining_columns if "year" in column.lower()), "None")
            year_column = st.selectbox("Year column (optional)", options=year_options, index=year_options.index(year_guess), key="trend_year_column", help="Without a year, matching calendar dates are combined into a repeating calendar pattern.")
            day_options = ["None"] + [column for column in remaining_columns if column != year_column]
            if st.session_state.get("trend_day_column") not in day_options:
                st.session_state.pop("trend_day_column", None)
            day_guess = next((column for column in day_options if column != "None" and "day" in column.lower()), "None")
            day_column = st.selectbox("Day column (optional)", options=day_options, index=day_options.index(day_guess), key="trend_day_column")

            month_quoted = "`" + month_column.replace("`", "``") + "`"
            month_text = F.lower(F.trim(F.col(month_quoted).cast("string")))
            month_names = {
                "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
                "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
                "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
                "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
            }
            month_lookup = F.create_map(*[item for pair in month_names.items() for item in (F.lit(pair[0]), F.lit(pair[1]))])
            month_number = F.coalesce(F.expr(f"try_cast({month_quoted} AS INT)"), month_lookup[month_text])
            year_quoted = "`" + year_column.replace("`", "``") + "`" if year_column != "None" else None
            day_quoted = "`" + day_column.replace("`", "``") + "`" if day_column != "None" else None
            year_number = F.expr(f"try_cast({year_quoted} AS INT)") if year_quoted else F.lit(2000)
            day_number = F.expr(f"try_cast({day_quoted} AS INT)") if day_quoted else F.lit(1)
            date_part_columns = {month_column}
            if year_column != "None":
                date_part_columns.add(year_column)
            if day_column != "None":
                date_part_columns.add(day_column)
            date_text = F.concat_ws(
                "-",
                year_number.cast("string"),
                F.lpad(month_number.cast("string"), 2, "0"),
                F.lpad(day_number.cast("string"), 2, "0"),
            )
            parsed_date = F.try_to_timestamp(date_text, F.lit("yyyy-MM-dd"))
            date_column = f"{month_column}" + (f" + {day_column}" if day_column != "None" else "") + (f" + {year_column}" if year_column != "None" else "")
            has_calendar_year = year_column != "None"
            st.caption("Month values can be numbers (1–12) or names such as Jan or January.")
            selected_parts = [month_column] + ([day_column] if day_column != "None" else []) + ([year_column] if year_column != "None" else [])
            part_preview = df.select(*[F.col("`" + column.replace("`", "``") + "`") for column in selected_parts]).limit(5).toPandas()
            with st.expander("Preview selected date parts"):
                display_table(part_preview)
        else:
            date_column = st.selectbox("Date/time column", options=temporal_candidates, key="trend_date_column")
            date_field_type = next(field.dataType.simpleString() for field in df.schema.fields if field.name == date_column)
            quoted_date = "`" + date_column.replace("`", "``") + "`"
            date_value = F.col(quoted_date)
            has_calendar_year = True
            if date_field_type == "string":
                date_formats = {
                    "Auto-detect": None,
                    "YYYY-MM-DD": "yyyy-MM-dd",
                    "YYYY-MM-DD HH:MM:SS": "yyyy-MM-dd HH:mm:ss",
                    "YYYY-MM-DDTHH:MM:SS": "yyyy-MM-dd'T'HH:mm:ss",
                    "YYYY-MM-DDTHH:MM:SS.SSS": "yyyy-MM-dd'T'HH:mm:ss.SSS",
                    "MM/DD/YYYY": "MM/dd/yyyy",
                    "DD/MM/YYYY": "dd/MM/yyyy",
                    "MM-DD-YYYY": "MM-dd-yyyy",
                    "DD-MM-YYYY": "dd-MM-yyyy",
                    "YYYY/MM/DD": "yyyy/MM/dd",
                    "YYYYMMDD": "yyyyMMdd",
                    "MM/DD/YYYY HH:MM:SS": "MM/dd/yyyy HH:mm:ss",
                    "DD-MMM-YYYY": "dd-MMM-yyyy",
                    "MMM D, YYYY": "MMM d, yyyy",
                    "Custom date pattern": "custom",
                }
                format_label = st.selectbox("Date format", options=list(date_formats), key="trend_date_format", help="Use Auto-detect first. If dates do not parse, select a listed format or enter a custom one.")
                date_format = date_formats[format_label]
                if date_format == "custom":
                    date_format = st.text_input("Custom date pattern", placeholder="Example: dd.MM.yyyy", key="trend_custom_date_format").strip()
                    parsed_date = F.try_to_timestamp(date_value, F.lit(date_format)) if date_format else F.lit(None).cast("timestamp")
                elif date_format is None:
                    common_unambiguous_formats = [
                        "yyyy-MM-dd HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss.SSS",
                        "yyyyMMdd",
                        "yyyy/MM/dd",
                        "dd-MMM-yyyy",
                        "MMM d, yyyy",
                    ]
                    parsed_date = F.coalesce(
                        F.try_to_timestamp(date_value),
                        *[F.try_to_timestamp(date_value, F.lit(pattern)) for pattern in common_unambiguous_formats],
                    )
                else:
                    parsed_date = F.try_to_timestamp(date_value, F.lit(date_format))
                date_preview = df.select(date_value.alias("Sample value")).where(date_value.isNotNull()).limit(5).toPandas()
                with st.expander("Preview values from this column"):
                    display_table(date_preview)
            else:
                parsed_date = F.to_timestamp(date_value)

        measure_options = ["Record count"] + list(info["numeric"])
        measure_choice = st.selectbox("What to trend", options=measure_options, key="trend_measure")
        if measure_choice == "Record count":
            aggregation = "Count"
            measure_column = None
        else:
            aggregation = st.selectbox("Measure calculation", options=["Sum", "Average"], key="trend_aggregation", help="Sum fits transaction amounts; Average can be more useful for per-account balances. Choose based on what one row represents.")
            quoted_measure = "`" + measure_choice.replace("`", "``") + "`"
            measure_column = F.col(quoted_measure)

        group_options = ["None"] + [column for column in info.get("categorical", []) if column != date_column and column not in date_part_columns]
        if st.session_state.get("trend_group_by") not in group_options:
            st.session_state.pop("trend_group_by", None)
        group_by = st.selectbox("Split trend by category (optional)", options=group_options, key="trend_group_by")
        if date_mode == "Separate date parts":
            if not has_calendar_year:
                st.caption("No year is selected; records sharing the same month/day are combined into a repeating calendar pattern.")
                granularity_options = ["Day", "Week", "Month", "Quarter"] if day_column != "None" else ["Month", "Quarter"]
                granularity_label = "Calendar interval"
                granularity_default = "Day" if day_column != "None" else "Month"
            else:
                granularity_options = ["Day", "Week", "Month", "Quarter", "Year"] if day_column != "None" else ["Month", "Quarter", "Year"]
                granularity_label = "Time interval"
                granularity_default = "Month"
            if st.session_state.get("trend_granularity") not in granularity_options:
                st.session_state.pop("trend_granularity", None)
            granularity = st.selectbox(granularity_label, options=granularity_options, index=granularity_options.index(granularity_default), key="trend_granularity")
            month_only_pattern = not has_calendar_year
        else:
            if st.session_state.get("trend_granularity") not in ["Day", "Week", "Month", "Quarter", "Year"]:
                st.session_state.pop("trend_granularity", None)
            granularity = st.selectbox("Time interval", options=["Day", "Week", "Month", "Quarter", "Year"], index=2, key="trend_granularity")
            month_only_pattern = False
        granularity_key = {"Day": "day", "Week": "week", "Month": "month", "Quarter": "quarter", "Year": "year"}[granularity]

        trend_data = df.withColumn("__trend_time", parsed_date).where(F.col("__trend_time").isNotNull())
        valid_date_rows = trend_data.count()
        report_sections.append(("Trend settings", pd.DataFrame([{
            "Date fields": date_column,
            "Date mode": date_mode,
            "Measure": measure_choice,
            "Calculation": aggregation,
            "Interval": granularity,
            "Category split": group_by,
            "Rows with parsed dates": valid_date_rows,
        }])))
        if valid_date_rows == 0:
            st.warning("No values in this column could be read as dates. Check the sample values above, choose a matching format, or enter a custom date pattern.")
            report_sections.append(("Trend result", "No rows could be interpreted as dates using the selected date fields."))
        else:
            period_column = F.date_trunc(granularity_key, F.col("__trend_time")).alias("Period")
            if measure_column is None:
                trend_aggregate = F.count(F.lit(1)).alias("Value")
                value_label = "Records"
            elif aggregation == "Sum":
                trend_aggregate = F.sum(measure_column).alias("Value")
                value_label = f"Sum of {measure_choice}"
            else:
                trend_aggregate = F.avg(measure_column).alias("Value")
                value_label = f"Average {measure_choice}"

            st.metric("Rows with a parsed date", f"{valid_date_rows:,}", help=f"{valid_date_rows:,} of {info['rows']:,} uploaded rows had a usable value in {date_column}.")
            if group_by == "None":
                trends = trend_data.groupBy(period_column).agg(trend_aggregate).orderBy("Period")
                series_label = None
            else:
                quoted_group = "`" + group_by.replace("`", "``") + "`"
                category_value = F.coalesce(F.col(quoted_group).cast("string"), F.lit("Missing"))
                top_categories = [row[0] for row in trend_data.groupBy(category_value.alias("__trend_category")).count().orderBy(F.col("count").desc()).limit(8).collect()]
                trend_data = trend_data.withColumn(
                    "__trend_series",
                    F.when(category_value.isin(top_categories), category_value).otherwise(F.lit("Other values")),
                )
                trends = (
                    trend_data.groupBy(period_column, F.col("__trend_series").alias("Series"))
                    .agg(trend_aggregate)
                    .orderBy("Period", "Series")
                )
                series_label = "Series"
                if trend_data.select("__trend_series").distinct().count() > len(top_categories):
                    st.caption("The chart keeps the eight most common categories separate and combines the rest as ‘Other values’.")

            period_count = trends.select("Period").distinct().count()
            if period_count > 5000:
                st.warning("This selection creates more than 5,000 time periods. Choose a wider interval to keep the chart responsive.")
                report_sections.append(("Trend result", "More than 5,000 time periods were generated. Choose a wider interval to keep the chart responsive."))
            else:
                trends_pd = trends.toPandas()
                if not trends_pd.empty:
                    trend_chart_options = {}
                    if month_only_pattern:
                        period_values = pd.to_datetime(trends_pd["Period"])
                        if granularity == "Month":
                            trends_pd["Period"] = period_values.dt.strftime("%b")
                        elif granularity == "Quarter":
                            trends_pd["Period"] = "Q" + period_values.dt.quarter.astype(str)
                        else:
                            trends_pd["Period"] = period_values.dt.strftime("%b %d")
                        trend_chart_options["category_orders"] = {"Period": list(dict.fromkeys(trends_pd["Period"].tolist()))}
                    st.markdown(f"#### {value_label} by {granularity.lower()}")
                    trend_fig = px.line(
                        trends_pd,
                        x="Period",
                        y="Value",
                        color=series_label,
                        markers=period_count <= 60,
                        title=f"{value_label} over time" + (f" by {group_by}" if series_label else ""),
                        **trend_chart_options,
                    )
                    trend_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color": text, "family": "Inter, Segoe UI, sans-serif"}, title_font={"color": text, "size": 16}, legend_title_text=group_by if series_label else None, margin={"l": 16, "r": 12, "t": 48, "b": 16}, hovermode="x unified")
                    trend_fig.update_xaxes(showgrid=False, linecolor=border, tickfont_color=muted, title_text="Calendar date" if month_only_pattern else "Date")
                    trend_fig.update_yaxes(gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_text=value_label)
                    st.plotly_chart(trend_fig, use_container_width=True)
                    report_sections.append((f"{value_label} over time chart", trend_fig))
                    display_table(trends_pd.head(1000))
                    report_sections.append(("Aggregated time trend", trends_pd))
                    if len(trends_pd) > 1000:
                        st.caption("The table shows the first 1,000 time-series rows. The chart includes all plotted periods.")
                    trend_csv = trends_pd.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("Download time trend (CSV)", data=trend_csv, file_name="time_trends.csv", mime="text/csv")

elif workspace_page == "Clustering":
    defaults = info["numeric"] or df.columns
    st.markdown("#### Clustering features")
    features = st.multiselect("Features to cluster on", options=df.columns, default=defaults, key="cluster_features_input", help="Pick fields that describe the observations. Exclude IDs, names, dates, and direct labels unless there is a clear analytical reason to include them.")
    outcome_options = ["None"] + list(df.columns)
    outcome_default = outcome_options.index(st.session_state["outcome_column"]) if st.session_state.get("outcome_column") in df.columns else 0
    outcome_choice = st.selectbox(
        "Outcome column (optional)",
        options=outcome_options,
        index=outcome_default,
        key="outcome_column",
        help="Choose a result or label to describe after clustering. It is excluded from K-Means so it cannot influence the groups.",
    )
    outcome_column = None if outcome_choice == "None" else outcome_choice
    model_features = [feature for feature in features if feature != outcome_column]
    if outcome_column in features:
        st.info(f"‘{outcome_column}’ is selected as the outcome, so it will be left out of the clustering features.")
    if outcome_column:
        st.caption(f"Outcome selected: **{outcome_column}**. It will be summarized by cluster after K-Means runs.")

    if clustering_page == "Compare K values":
        st.markdown("#### Compare candidate cluster counts")
        if info["rows"] < 3:
            st.info("At least three records are needed to compare multiple K values.")
        else:
            compare_upper = min(8, info["rows"])
            compare_max = st.slider(
                "Compare K from 2 through",
                min_value=3,
                max_value=compare_upper,
                value=min(5, compare_upper),
                key="compare_max_k",
            )
            comparison_key = (tuple(model_features), compare_max)
            if st.session_state.get("k_comparison_key") not in (None, comparison_key):
                st.session_state.pop("k_comparison", None)
                st.session_state.pop("k_comparison_key", None)
            st.caption("Spark fits a separate model for each K in the range. This can take a little time on a large dataset.")
            if st.button("Compare K values", key="compare_k_values", type="primary", disabled=not model_features):
                try:
                    with st.spinner(f"Comparing K=2 through K={compare_max} with Spark…"):
                        scores = []
                        for candidate_k in range(2, compare_max + 1):
                            _, candidate_score = cluster_dataset(df, model_features, candidate_k)
                            scores.append({"K": candidate_k, "Silhouette": candidate_score})
                    st.session_state["k_comparison"] = scores
                    st.session_state["k_comparison_key"] = comparison_key
                except Exception as exc:
                    st.error(f"K comparison could not be completed: {exc}")

            if st.session_state.get("k_comparison_key") == comparison_key:
                comparison = pd.DataFrame(st.session_state["k_comparison"])
                best_row = comparison.loc[comparison["Silhouette"].idxmax()]
                best_k = int(best_row["K"])
                best_score = float(best_row["Silhouette"])
                score_col, use_col = st.columns([1, 2])
                score_col.metric("Highest score in this range", f"{best_score:.3f}", help="A higher silhouette score is generally preferable, but also check cluster balance and interpretability.")
                if best_k != st.session_state.get("selected_k", min(3, max(2, min(10, info["rows"])) )):
                    use_col.button(
                        f"Use K = {best_k} in Run K-Means",
                        key="use_compared_k",
                        on_click=select_recommended_k,
                        args=(best_k,),
                    )
                compare_fig = px.line(comparison, x="K", y="Silhouette", markers=True, title="Silhouette by cluster count")
                compare_fig.update_traces(line={"color":"#31b9cf","width":3}, marker={"color":"#9a6cff","size":10})
                compare_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color":text,"family":"Inter, Segoe UI, sans-serif"}, title_font={"color":text,"size":16}, margin={"l":16,"r":12,"t":48,"b":16})
                compare_fig.update_xaxes(dtick=1, showgrid=False, linecolor=border, tickfont_color=muted)
                compare_fig.update_yaxes(gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_font_color=muted)
                st.plotly_chart(compare_fig, use_container_width=True)
                report_sections.append(("Silhouette comparison chart", compare_fig))
                display_table(comparison)
                report_sections.append(("K comparison scores", comparison))
                st.caption("This is the highest score among the K values tested. Review cluster profiles and size balance on the Run K-Means page before choosing K.")

    else:
        st.markdown("#### Run K-Means")
        max_k = max(2, min(10, info["rows"]))
        k = st.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=min(3,max_k), key="selected_k")
        if "cluster_result" in st.session_state and (
            tuple(st.session_state.get("cluster_features", ())) != tuple(model_features)
            or st.session_state.get("cluster_k") != k
        ):
            st.session_state.pop("cluster_result", None)
            st.session_state.pop("cluster_score", None)
            st.session_state.pop("segment_drilldown_cluster", None)
            st.session_state.pop("segment_category_feature", None)
            st.session_state.pop("segment_numeric_feature", None)
        if st.button("Run K-Means with Spark", type="primary", disabled=info["rows"] < 2 or len(model_features) == 0):
            with st.spinner("Preparing features and fitting Spark MLlib K-Means…"):
                try:
                    result, score = cluster_dataset(df, model_features, k)
                    st.session_state.cluster_result = result
                    st.session_state.cluster_score = score
                    st.session_state.cluster_k = k
                    st.session_state.cluster_features = tuple(model_features)
                    st.session_state.cluster_outcome = outcome_column
                except Exception as exc:
                    st.error(f"Clustering could not be completed: {exc}")
        if "cluster_result" in st.session_state:
            result = st.session_state.cluster_result
            score = st.session_state.cluster_score
            st.metric("Silhouette score", f"{score:.3f}", help="Values closer to 1 generally indicate more distinct clusters; interpret this alongside cluster sizes and the chosen features.")
            summary = result.groupBy("prediction").count().orderBy("prediction").toPandas().rename(columns={"prediction":"Cluster", "count":"Records"})
            report_sections.append(("K-Means evaluation", pd.DataFrame([{
                "Clusters (K)": st.session_state.get("cluster_k", k),
                "Silhouette score": round(float(score), 4),
                "Features": ", ".join(st.session_state.get("cluster_features", tuple(model_features))),
            }])))
            report_sections.append(("Records per cluster", summary))
            left,right=st.columns([1,1])
            with left:
                bar_fig = px.bar(summary, x="Cluster", y="Records", color="Cluster", title="Records per cluster", color_discrete_sequence=["#31b9cf","#9a6cff","#42c6a5","#efb750","#ed7b8e"])
                bar_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color":text,"family":"Inter, Segoe UI, sans-serif"}, title_font={"color":text,"size":16}, legend_title_text="Segment", margin={"l":16,"r":12,"t":48,"b":16})
                bar_fig.update_xaxes(showgrid=False, linecolor=border, tickfont_color=muted)
                bar_fig.update_yaxes(gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_font_color=muted)
                st.plotly_chart(bar_fig, use_container_width=True)
                report_sections.append(("Records per cluster chart", bar_fig))
            with right:
                pie_fig = px.pie(summary, names="Cluster", values="Records", hole=.68, title="Cluster distribution", color_discrete_sequence=["#31b9cf","#9a6cff","#42c6a5","#efb750","#ed7b8e"])
                pie_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color":text,"family":"Inter, Segoe UI, sans-serif"}, title_font={"color":text,"size":16}, legend_title_text="Segment", margin={"l":12,"r":12,"t":48,"b":16})
                pie_fig.update_traces(textfont_color="#f4f7fb", marker_line_color=panel, marker_line_width=2)
                st.plotly_chart(pie_fig, use_container_width=True)
                report_sections.append(("Cluster distribution chart", pie_fig))
            st.markdown("#### Cluster profiles")
            st.caption("Numeric features show their cluster average; categorical features show their most common value.")
            model_features = st.session_state.get("cluster_features", tuple(features))
            profile_aggs = [F.count(F.lit(1)).alias("Records")]
            numeric_features = set(info["numeric"])
            for feature in model_features:
                quoted_feature = "`" + feature.replace("`", "``") + "`"
                column = F.col(quoted_feature)
                if feature in numeric_features:
                    profile_aggs.append(F.round(F.avg(column), 2).alias(f"Average · {feature}"))
                else:
                    profile_aggs.append(F.mode(column).alias(f"Most common · {feature}"))
            profile = result.groupBy("prediction").agg(*profile_aggs).orderBy("prediction").toPandas().rename(columns={"prediction":"Cluster"})
            display_table(profile)
            report_sections.append(("Cluster profiles", profile))
            if outcome_column:
                st.markdown("#### Outcome by cluster")
                st.caption(f"This describes **{outcome_column}** for the assigned groups. The outcome was excluded from the K-Means model.")
                quoted_outcome = "`" + outcome_column.replace("`", "``") + "`"
                outcome_value = F.col(quoted_outcome)
                if outcome_column in numeric_features:
                    outcome_summary = (
                        result.groupBy("prediction")
                        .agg(
                            F.count(F.lit(1)).alias("Records"),
                            F.count(outcome_value).alias("Non-missing outcomes"),
                            F.round(F.avg(outcome_value), 2).alias(f"Average · {outcome_column}"),
                        )
                        .orderBy("prediction")
                        .toPandas()
                        .rename(columns={"prediction": "Cluster"})
                    )
                    display_table(outcome_summary)
                    report_sections.append((f"Numeric outcome by cluster · {outcome_column}", outcome_summary))
                    average_column = f"Average · {outcome_column}"
                    chart_data = outcome_summary.dropna(subset=[average_column])
                    if not chart_data.empty:
                        outcome_fig = px.bar(
                            chart_data,
                            x="Cluster",
                            y=average_column,
                            color="Cluster",
                            title=f"Average {outcome_column} by cluster",
                            color_discrete_sequence=["#31b9cf", "#9a6cff", "#42c6a5", "#efb750", "#ed7b8e"],
                        )
                        outcome_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color": text, "family": "Inter, Segoe UI, sans-serif"}, title_font={"color": text, "size": 16}, showlegend=False, margin={"l": 16, "r": 12, "t": 48, "b": 16})
                        outcome_fig.update_xaxes(showgrid=False, linecolor=border, tickfont_color=muted)
                        outcome_fig.update_yaxes(gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_font_color=muted)
                        st.plotly_chart(outcome_fig, use_container_width=True)
                        report_sections.append((f"Average {outcome_column} by cluster chart", outcome_fig))
                else:
                    outcome_groups = (
                        result.groupBy("prediction", F.coalesce(outcome_value.cast("string"), F.lit("Missing")).alias("Outcome"))
                        .agg(F.count(F.lit(1)).alias("Records"))
                        .withColumnRenamed("prediction", "Cluster")
                    )
                    outcome_groups = outcome_groups.withColumn(
                        "Share (%)",
                        F.round(F.col("Records") * 100.0 / F.sum("Records").over(Window.partitionBy("Cluster")), 1),
                    ).orderBy("Cluster", "Outcome")
                    outcome_pd = outcome_groups.toPandas()
                    display_table(outcome_pd)
                    report_sections.append((f"Categorical outcome by cluster · {outcome_column}", outcome_pd))
                    if outcome_pd["Outcome"].nunique() <= 12:
                        outcome_fig = px.bar(
                            outcome_pd,
                            x="Cluster",
                            y="Share (%)",
                            color="Outcome",
                            barmode="stack",
                            title=f"{outcome_column} mix within each cluster",
                        )
                        outcome_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color": text, "family": "Inter, Segoe UI, sans-serif"}, title_font={"color": text, "size": 16}, legend_title_text=outcome_column, margin={"l": 16, "r": 12, "t": 48, "b": 16})
                        outcome_fig.update_xaxes(showgrid=False, linecolor=border, tickfont_color=muted)
                        outcome_fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_font_color=muted)
                        st.plotly_chart(outcome_fig, use_container_width=True)
                        report_sections.append((f"{outcome_column} mix by cluster chart", outcome_fig))
                    else:
                        st.caption("The table lists all outcome values. The chart is hidden because this column has more than 12 distinct values.")
            st.markdown("#### Segment drill-down")
            segment_records = summary.set_index("Cluster")["Records"].to_dict()
            segment_ids = list(segment_records)
            selected_segment = st.selectbox(
                "Choose a cluster to inspect",
                options=segment_ids,
                format_func=lambda cluster_id: f"Cluster {cluster_id} · {segment_records[cluster_id]:,} records",
                key="segment_drilldown_cluster",
            )
            segment = result.filter(F.col("prediction") == selected_segment)
            selected_count = int(segment_records[selected_segment])
            segment_share = selected_count / max(int(info["rows"]), 1) * 100
            drill_metrics = st.columns(3)
            drill_metrics[0].metric("Records in cluster", f"{selected_count:,}")
            drill_metrics[1].metric("Share of dataset", f"{segment_share:.1f}%")
            drill_metrics[2].metric("Features used", len(model_features))
            selected_profile = profile.loc[profile["Cluster"] == selected_segment]
            st.markdown("##### This cluster's profile")
            display_table(selected_profile)
            report_sections.append((f"Selected segment · Cluster {selected_segment}", selected_profile))

            categorical_model_features = [feature for feature in model_features if feature not in numeric_features]
            numeric_model_features = [feature for feature in model_features if feature in numeric_features]
            if categorical_model_features:
                selected_category_feature = st.selectbox(
                    "Explore a categorical feature",
                    options=categorical_model_features,
                    key="segment_category_feature",
                )
                quoted_category_feature = "`" + selected_category_feature.replace("`", "``") + "`"
                category_breakdown = (
                    segment.groupBy(F.coalesce(F.col(quoted_category_feature).cast("string"), F.lit("Missing")).alias("Value"))
                    .agg(F.count(F.lit(1)).alias("Records"))
                    .withColumn("Share (%)", F.round(F.col("Records") * 100.0 / F.lit(selected_count), 1))
                    .orderBy(F.col("Records").desc(), F.col("Value"))
                    .limit(15)
                )
                category_pd = category_breakdown.toPandas()
                st.markdown(f"##### {selected_category_feature} in Cluster {selected_segment}")
                display_table(category_pd)
                report_sections.append((f"Cluster {selected_segment} · {selected_category_feature} breakdown", category_pd))
                if not category_pd.empty:
                    category_fig = px.bar(category_pd, x="Share (%)", y="Value", orientation="h", title=f"Most common {selected_category_feature} values", color="Share (%)", color_continuous_scale=["#31b9cf", "#9a6cff"])
                    category_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color": text, "family": "Inter, Segoe UI, sans-serif"}, title_font={"color": text, "size": 16}, coloraxis_showscale=False, margin={"l": 16, "r": 12, "t": 48, "b": 16})
                    category_fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=border, tickfont_color=muted, title_font_color=muted)
                    category_fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_color=muted)
                    st.plotly_chart(category_fig, use_container_width=True)
                    report_sections.append((f"Cluster {selected_segment} · {selected_category_feature} chart", category_fig))

            if numeric_model_features:
                selected_numeric_feature = st.selectbox(
                    "Explore a numeric feature",
                    options=numeric_model_features,
                    key="segment_numeric_feature",
                )
                quoted_numeric_feature = "`" + selected_numeric_feature.replace("`", "``") + "`"
                numeric_sample = (
                    segment.select(F.col(quoted_numeric_feature).alias("Value"))
                    .where(F.col("Value").isNotNull())
                    .sample(withReplacement=False, fraction=min(1.0, 5000 / max(selected_count, 1)), seed=42)
                    .limit(5000)
                    .toPandas()
                )
                st.markdown(f"##### {selected_numeric_feature} distribution in Cluster {selected_segment}")
                if not numeric_sample.empty:
                    numeric_fig = px.histogram(numeric_sample, x="Value", nbins=30, title=f"{selected_numeric_feature} distribution", color_discrete_sequence=["#31b9cf"])
                    numeric_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color": text, "family": "Inter, Segoe UI, sans-serif"}, title_font={"color": text, "size": 16}, showlegend=False, margin={"l": 16, "r": 12, "t": 48, "b": 16})
                    numeric_fig.update_xaxes(gridcolor=border, tickfont_color=muted)
                    numeric_fig.update_yaxes(gridcolor=border, tickfont_color=muted, title_text="Records")
                    st.plotly_chart(numeric_fig, use_container_width=True)
                    report_sections.append((f"Cluster {selected_segment} · {selected_numeric_feature} distribution chart", numeric_fig))
                    if selected_count > 5000:
                        st.caption("The histogram uses a random sample of up to 5,000 records for responsiveness.")

            st.markdown("##### Records in this cluster")
            display_table(segment.limit(100).toPandas())
            segment_csv = segment.toPandas().to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"Download Cluster {selected_segment} records (CSV)",
                data=segment_csv,
                file_name=f"cluster_{selected_segment}_records.csv",
                mime="text/csv",
                key="download_selected_cluster",
            )
            st.caption("The table previews up to 100 rows. The download contains all records assigned to this cluster.")
            display_table(result.limit(100).toPandas())
            csv_bytes = result.toPandas().to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download clustered records (CSV)", data=csv_bytes, file_name="clustered_records.csv", mime="text/csv")
            st.caption("The silhouette score is calculated with Spark MLlib's squared Euclidean distance. K-Means is sensitive to feature choice and outliers; review feature meaning and cluster balance before interpreting the results.")

elif workspace_page == "Spark SQL":
    st.markdown("#### Spark SQL summary")
    st.caption("The app registers the uploaded data as a temporary Spark SQL view. Table and column identifiers are quoted from the detected schema.")
    view_name = "uploaded_dataset"
    df.createOrReplaceTempView(view_name)
    count_sql = f"SELECT COUNT(*) AS records FROM `{view_name}`"
    st.code(count_sql + ";", language="sql")
    sql_count = spark.sql(count_sql).toPandas()
    display_table(sql_count)
    report_sections.append(("Spark SQL record count", sql_count))
    if info["numeric"]:
        picked = st.selectbox("Numeric column for SQL profile", info["numeric"])
        quoted_col = "`" + picked.replace("`", "``") + "`"
        profile_sql = f"SELECT COUNT({quoted_col}) AS non_null, AVG({quoted_col}) AS mean, MIN({quoted_col}) AS minimum, MAX({quoted_col}) AS maximum FROM `{view_name}`"
        st.code(profile_sql + ";", language="sql")
        sql_profile = spark.sql(profile_sql)
        sql_profile_pd = sql_profile.toPandas()
        display_table(sql_profile_pd)
        report_sections.append((f"Spark SQL profile · {picked}", sql_profile_pd))

    st.markdown("#### Run your own Spark SQL query")
    st.caption("Enter a read-only SELECT query against `uploaded_dataset`. Results are limited to 1,000 preview rows.")
    st.text_area(
        "SQL query",
        value=f"SELECT * FROM `{view_name}` LIMIT 20",
        key="sql_editor",
        height=130,
        label_visibility="collapsed",
    )
    if st.button("Run SQL query", key="run_sql_query", type="primary"):
        query = st.session_state.sql_editor.strip()
        if query.endswith(";"):
            query = query[:-1].rstrip()
        first_word = query.split(None, 1)[0].upper() if query else ""
        if first_word not in {"SELECT", "WITH"}:
            st.error("Only read-only SELECT queries (including WITH queries) are allowed.")
        elif ";" in query:
            st.error("Run one SQL statement at a time; remove internal semicolons.")
        else:
            try:
                query_result = spark.sql(query)
                display_table(query_result.limit(1000).toPandas())
                st.caption("Showing at most 1,000 rows. Use LIMIT or an aggregate to keep previews manageable.")
            except Exception as exc:
                st.error(f"Spark SQL could not run this query: {exc}")

st.divider()
st.markdown("#### Download analysis report")
st.caption("The self-contained HTML report includes dataset quality, detected schema, and aggregate results from this page. Open it in a browser and print or save it as PDF if needed.")
report_html = build_analysis_report(uploaded.name, uploaded.size, workspace_page, info, schema_report, report_sections)
safe_report_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded.name).stem).strip("_") or "dataset"
st.download_button(
    "Download analysis report (HTML)",
    data=report_html.encode("utf-8"),
    file_name=f"finsight_{safe_report_name}_analysis_report.html",
    mime="text/html",
    type="primary",
    key="download_analysis_report",
)
