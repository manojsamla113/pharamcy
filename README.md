# 💊 PharmaDash

> Multi-store pharmacy analytics dashboard for Hyderabad pharmacy chains — built with Python, Streamlit, and Groq AI.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit) ![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?logo=plotly) ![Groq](https://img.shields.io/badge/AI-Groq%20Llama%203.3-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

PharmaDash is a full-stack pharmacy analytics platform that centralises inventory, sales, supplier, and financial data across **5 Hyderabad pharmacy stores** into a single interactive web dashboard. It enables data-driven decisions on stock management, expiry tracking, reorder planning, supplier evaluation, and revenue analysis — with an AI-powered Q&A assistant built on Groq's Llama-3.3-70B.

**Current version: v2.0** (`pharmacy_updated.py`) — 3,083 lines, +52% from v1.

---

## Stores Covered

| Store | Location |
|---|---|
| MedPlus | Ameerpet, Hyderabad |
| Apollo | Kukatpally, Hyderabad |
| WellCare | Banjara Hills, Hyderabad |
| HealthFirst | Madhapur, Hyderabad |
| CityMed | Dilsukhnagar, Hyderabad |

---

## Features

### Dashboard Pages
- **Home** — Executive KPI snapshot and platform overview
- **Overview** — Revenue, inventory value, expiry health, net profit estimate, GST estimate, Rx vs OTC split
- **Inventory & Expiry** — Batch-level expiry tracking with conditional colouring, expiry timeline histogram
- **Sales & Demand** — Monthly trends, top medicines, category splits, medicine velocity (fast/slow moving)
- **Supplier & Store** — Interactive OpenStreetMap store bubbles, supplier risk scoring, quality scatter plot
- **Business Insights** — Store underperformance alerts, below-cost category flags, auto-generated reorder list
- **AI Assistant** — Multi-turn Groq-powered chat with pre-built insight cards and bubble-style history
- **About** — Platform documentation and tech stack

### Key Capabilities
- Real-time sidebar filters (date range, store, category, supplier, stock status) — all pages stay in sync
- Composite supplier **RiskScore** = `OOSBatches×10 + LowStock×3 + Expiring×1 + delay×0.3 + lead×0.5 + quality×2`
- Accurate out-of-stock counts deduped on `[Store_Name, ProductID]` (93+ unique SKU+store combinations)
- Suggested reorder quantities: `max(1, round(ReorderPoint × 1.5 − QuantityOnHand))`
- Excel report export from any filtered view
- Glassmorphism UI — no external CSS file, all styles inline

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit ≥ 1.32 |
| Data | Pandas ≥ 2.0, NumPy ≥ 1.26 |
| Charts | Plotly Express / Graph Objects |
| Maps | OpenStreetMap via Plotly scatter_mapbox |
| Excel Export | openpyxl ≥ 3.1 |
| AI | Groq API — Llama-3.3-70B |
| Config | python-dotenv |

---

## Project Structure

```
pharma_dash/
├── pharmacy_updated.py   # Main application (v2.0)
├── requirements.txt
├── .env                  # API key (not committed — see setup)
└── dataset/
    ├── Fact_Sales_20k.csv
    ├── Fact_Stock_20k.csv
    ├── Dim_Products_20k.csv
    ├── Dim_Store.csv
    └── Dim_Suppliers_20k.csv
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Manideep0704/pharma-dash.git
cd pharma-dash
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Configure dataset path

In `pharmacy_updated.py`, update the `base` path to point to your `dataset/` folder:

```python
# Replace the hardcoded path with:
base = os.environ.get(
    "PHARMA_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
)
```

Place all five CSV files in the `dataset/` subdirectory.

### 5. Run

```bash
streamlit run pharmacy_updated.py
```

App opens at **http://localhost:8501**

---

## Data Model

| File | Rows | Description |
|---|---|---|
| `Fact_Sales_20k.csv` | 20,000 | Sales transactions (date, store, product, revenue) |
| `Fact_Stock_20k.csv` | 20,000 | Stock batches (expiry, quantity, reorder levels) |
| `Dim_Products_20k.csv` | — | Product master (prices, reorder thresholds) |
| `Dim_Store.csv` | 5 | Store names and IDs |
| `Dim_Suppliers_20k.csv` | — | Supplier names, quality ratings, lead times |

All joins are performed once at load time via `@st.cache_data` and never mutated.

---

## Performance Notes

- All data loaded once via `@st.cache_data` — zero re-reads on filter changes
- Numeric columns downcasted to `float32` (Pandas 2.x safe: `pd.to_numeric(..., errors='coerce').astype('float32')`)
- High-cardinality string columns use `category` dtype for fast `groupby`
- KPIs computed with NumPy vectorised ops (`np.dot`, `np.select`) — no row-wise `apply()`
- `backdrop-filter: blur()` removed from all cards — eliminates GPU compositing jank on lower-end hardware
- AI responses cached by `(summary_data, user_question, chat_history)` — identical queries cost zero API calls

---

## Known Issues

| Issue | Status |
|---|---|
| Hardcoded Windows dataset path (`C:\Users\Asus\...`) | Fix by updating `base` path — see step 4 above |
| Dataset has only 6 medicine name templates across 170 ProductIDs | v2 bug fixes use `ProductID` for uniqueness — resolved for this dataset |
| GST fixed at 12% | Real pharmacy GST varies by category (0–18%) |
| Net Profit excludes operating expenses | Estimates only; does not include rent, salaries, utilities |
| No authentication layer | Do not expose publicly without adding auth |

---

## Changelog

### v2.0 (May 2026)
- Multi-turn AI chat with bubble UI and Clear Chat button
- New **Business Insights** page: underperformance alerts, below-cost flags, reorder list
- Bug fixes: `get_expiring_soon_n`, `get_out_of_stock_by_store`, `get_supplier_last_delivery_days`
- New `get_reorder_list()` helper with suggested order quantities
- Composite supplier RiskScore replacing simple delivery-day sort
- Enriched AI context: below-cost categories, safety stock KPIs, profitability estimate
- Pandas 2.x-safe dtype downcast
- Removed `backdrop-filter: blur()` for render performance
- Pre-flight CSV file validation with `st.error()` + `st.stop()`
- +1,052 lines (+52% from v1)

### v1.0
- Initial release with 7 dashboard pages and single-turn AI assistant

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*PharmaDash Analytics — v2.0 | May 2026*
