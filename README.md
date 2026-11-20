# Finance and Banking Clustering Analysis Using Spark SQL

A student-friendly local project for exploring financial and banking **tabular** datasets and segmenting their records with Spark MLlib K-Means. Upload any suitable CSV or Parquet schema; the application detects the columns and types rather than relying on a bundled dataset. Use customer-level data for customer segments. If you supply transaction-level data, each row is treated as a transaction.

## Features

- CSV and Parquet uploads with detected schema, sample rows, missing values, and duplicate counts.
- User-selected numeric and categorical features. Numeric gaps use median imputation; categories are indexed and one-hot encoded; assembled vectors are standardized.
- Spark SQL row counts and a numeric-column profile.
- Spark MLlib K-Means with a user-selected K and silhouette evaluation.
- Cluster distribution charts and CSV export.
- A step-by-step notebook and an approachable premium fintech dashboard.

No database, API, cloud service, authentication, streaming, or deep-learning components are used. Data processing runs locally. Upload only manageable files on a 12 GB RAM machine; large files may take time and use substantial memory.

## Requirements

- Python 3.10–3.14.
- Java 17 or 21.
- Windows, macOS, or Linux.

## Windows setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py`.

## Notebook

Open `notebooks/finance_banking_clustering.ipynb` in JupyterLab. Upload or copy a dataset into `data/uploads/`, then set the path in the notebook's first configuration cell. The notebook and dashboard share the same Spark pipeline.

## Data handling and interpretation

Feature selection is a modeling decision. Exclude IDs, direct labels, and fields that should not influence similarity. The app does not claim to verify that a dataset is financial: inspect the detected schema and choose a suitable dataset yourself. Silhouette is a diagnostic, not proof of useful business segments. Review cluster sizes and feature profiles before interpreting them.
