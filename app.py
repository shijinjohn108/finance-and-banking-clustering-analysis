"""FinSight: a local-first finance and banking clustering dashboard."""
from __future__ import annotations

import os
import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
        ["Explore data", "Clustering", "Spark SQL"],
        format_func=lambda page: {"Explore data": "▦   Explore data", "Clustering": "✳   Clustering", "Spark SQL": "⌘   Spark SQL"}[page],
        key="workspace_page",
        label_visibility="collapsed",
    )

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
        st.markdown('<div class="feature-card"><div class="feature-icon">K</div><h3>Find useful groups</h3><p>Build K-Means segments, review the silhouette score, and export results.</p></div>', unsafe_allow_html=True)
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

if workspace_page == "Explore data":
    st.markdown("#### Schema detected from your upload")
    schema_pd = pd.DataFrame([{"Column": f.name, "Detected type": f.dataType.simpleString(), "Missing": info["missing"].get(f.name, 0), "Suggested role": "Numeric" if f.name in info["numeric"] else "Categorical / identifier"} for f in df.schema.fields])
    display_table(schema_pd)
    st.markdown("#### Data preview")
    display_table(df.limit(100).toPandas())
    st.caption("No records are silently removed. Numeric missing values are imputed during modeling; categorical missing values are handled by Spark's invalid-category option.")

elif workspace_page == "Clustering":
    st.markdown("#### Configure a segmentation")
    defaults = info["numeric"] or df.columns
    features = st.multiselect("Features to cluster on", options=df.columns, default=defaults, help="Pick fields that describe the observations. Exclude IDs, names, dates, and direct labels unless there is a clear analytical reason to include them.")
    max_k = max(2, min(10, info["rows"]))
    k = st.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=min(3,max_k))
    if st.button("Run K-Means with Spark", type="primary", disabled=info["rows"] < 2 or len(features) == 0):
        with st.spinner("Preparing features and fitting Spark MLlib K-Means…"):
            try:
                result, score = cluster_dataset(df, features, k)
                st.session_state.cluster_result = result
                st.session_state.cluster_score = score
                st.session_state.cluster_k = k
            except Exception as exc:
                st.error(f"Clustering could not be completed: {exc}")
    if "cluster_result" in st.session_state:
        result = st.session_state.cluster_result
        score = st.session_state.cluster_score
        st.metric("Silhouette score", f"{score:.3f}", help="Values closer to 1 generally indicate more distinct clusters; interpret this alongside cluster sizes and the chosen features.")
        summary = result.groupBy("prediction").count().orderBy("prediction").toPandas().rename(columns={"prediction":"Cluster", "count":"Records"})
        left,right=st.columns([1,1])
        with left:
            bar_fig = px.bar(summary, x="Cluster", y="Records", color="Cluster", title="Records per cluster", color_discrete_sequence=["#31b9cf","#9a6cff","#42c6a5","#efb750","#ed7b8e"])
            bar_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color":text,"family":"Inter, Segoe UI, sans-serif"}, title_font={"color":text,"size":16}, legend_title_text="Segment", margin={"l":16,"r":12,"t":48,"b":16})
            bar_fig.update_xaxes(showgrid=False, linecolor=border, tickfont_color=muted)
            bar_fig.update_yaxes(gridcolor=border, zerolinecolor=border, tickfont_color=muted, title_font_color=muted)
            st.plotly_chart(bar_fig, use_container_width=True)
        with right:
            pie_fig = px.pie(summary, names="Cluster", values="Records", hole=.68, title="Cluster distribution", color_discrete_sequence=["#31b9cf","#9a6cff","#42c6a5","#efb750","#ed7b8e"])
            pie_fig.update_layout(paper_bgcolor=panel, plot_bgcolor=panel, font={"color":text,"family":"Inter, Segoe UI, sans-serif"}, title_font={"color":text,"size":16}, legend_title_text="Segment", margin={"l":12,"r":12,"t":48,"b":16})
            pie_fig.update_traces(textfont_color="#f4f7fb", marker_line_color=panel, marker_line_width=2)
            st.plotly_chart(pie_fig, use_container_width=True)
        display_table(result.limit(100).toPandas())
        csv_bytes = result.toPandas().to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download clustered records (CSV)", data=csv_bytes, file_name="clustered_records.csv", mime="text/csv")
        st.caption("The silhouette score is calculated with Spark MLlib's squared Euclidean distance. K-Means is sensitive to feature choice and outliers; compare a few K values before drawing conclusions.")

elif workspace_page == "Spark SQL":
    st.markdown("#### Spark SQL summary")
    st.caption("The app registers the uploaded data as a temporary Spark SQL view. Table and column identifiers are quoted from the detected schema.")
    view_name = "uploaded_dataset"
    df.createOrReplaceTempView(view_name)
    count_sql = f"SELECT COUNT(*) AS records FROM `{view_name}`"
    st.code(count_sql + ";", language="sql")
    display_table(spark.sql(count_sql).toPandas())
    if info["numeric"]:
        picked = st.selectbox("Numeric column for SQL profile", info["numeric"])
        quoted_col = "`" + picked.replace("`", "``") + "`"
        profile_sql = f"SELECT COUNT({quoted_col}) AS non_null, AVG({quoted_col}) AS mean, MIN({quoted_col}) AS minimum, MAX({quoted_col}) AS maximum FROM `{view_name}`"
        st.code(profile_sql + ";", language="sql")
        sql_profile = spark.sql(profile_sql)
        display_table(sql_profile.toPandas())

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
