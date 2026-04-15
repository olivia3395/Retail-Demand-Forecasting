<div align="center">

# 🛒 Promotion-Aware Retail Demand Forecasting with Transformers

### Industry-Style Multi-Horizon Forecasting for Store-SKU Demand Planning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyTorch Forecasting](https://img.shields.io/badge/PyTorch_Forecasting-TFT-6C63FF?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-Baseline-9ACD32?style=flat-square)
![Kaggle](https://img.shields.io/badge/Dataset-M5_Forecasting-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Forecasting](https://img.shields.io/badge/Task-Demand_Forecasting-F59E0B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-888780?style=flat-square)

**A complete, production-style demand forecasting project built on the M5 Forecasting dataset.**  
This repository downloads the data from Kaggle, preprocesses it into model-ready format, engineers retail covariates, benchmarks strong baselines, and trains a **Temporal Fusion Transformer (TFT)** for **28-day store-SKU demand forecasting**.

</div>



Retail demand forecasting is not just a time-series modeling exercise. In real business settings, planners care about:

- **store-SKU level demand**
- **promotion and pricing effects**
- **holiday and calendar shifts**
- **multi-horizon forecasting**
- **robust backtesting**
- **business-facing metrics for replenishment and inventory planning**

This project is designed to reflect that workflow. It is not a toy notebook. It is structured as a clean, reusable forecasting repo with automated data download, preprocessing, feature generation, model training, evaluation, and artifact export.



## What this project does

- Downloads the **M5 Forecasting - Accuracy** dataset from Kaggle
- Converts the raw wide sales table into a long-format forecasting dataset
- Builds retail forecasting features using:
  - prices
  - price deltas
  - SNAP flags
  - event indicators
  - weekday / month / seasonality signals
  - lag features
  - rolling statistics
  - lightweight promotion proxies
- Trains:
  - **Seasonal Naive**
  - **LightGBM**
  - **Temporal Fusion Transformer (TFT)**
- Evaluates with time-based validation using:
  - **MAE**
  - **RMSE**
  - **WAPE**
  - **sMAPE**
- Saves:
  - metrics
  - validation predictions
  - fitted models
  - processed datasets
  - TFT checkpoints and interpretation artifacts



## Business framing

**Goal:** forecast daily demand for each **store-SKU** over the next **28 days**.

**Use cases:**
- replenishment planning
- promotion-aware demand estimation
- stock risk monitoring
- category and store-level planning
- comparing high-volume versus long-tail product performance

This is the kind of problem that appears in retail, e-commerce, grocery, marketplace, and supply chain analytics teams.



## Dataset: M5 Forecasting - Accuracy

This project uses the **M5 Forecasting - Accuracy** competition dataset, a large-scale retail forecasting benchmark released on Kaggle. It is one of the most widely used public datasets for hierarchical demand forecasting.

### Dataset overview

The M5 dataset contains daily unit sales for Walmart products across multiple aggregation levels, including:

- **item level**
- **department level**
- **category level**
- **store level**
- **state level**

At the most granular level, the forecasting task is to predict future demand for each **item-store combination**.

### Core files used in this repo

#### `sales_train_validation.csv`
The main sales history table.

Key identifiers include:
- `item_id`
- `dept_id`
- `cat_id`
- `store_id`
- `state_id`

Target columns:
- `d_1, d_2, ..., d_n` representing daily unit sales

This is the source table that gets reshaped into long format.

#### `calendar.csv`
Calendar-level metadata for each day.

Useful fields include:
- date
- weekday
- month / year
- event names and event types
- SNAP indicators for CA / TX / WI

This is used to inject holiday, event, and seasonality structure into the model.

#### `sell_prices.csv`
Weekly sell prices for each store-item pair.

Useful fields include:
- `store_id`
- `item_id`
- `wm_yr_wk`
- `sell_price`

This table enables price-aware demand modeling and price-change feature construction.



## Project structure

```text
m5_tft_project/
├── configs/
│   └── default.yaml
├── scripts/
│   └── bootstrap.sh
├── src/
│   ├── data/
│   │   ├── download_m5.py
│   │   └── prepare_m5.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate_predictions.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── seasonal_naive.py
│   │   ├── train_lightgbm.py
│   │   └── train_tft.py
│   ├── utils/
│   │   ├── io.py
│   │   └── logging_utils.py
│   └── run_pipeline.py
├── requirements.txt
└── README.md
```


## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```



## Kaggle authentication

The M5 dataset is hosted as a Kaggle competition dataset, so authentication is required before download.

Choose one of the following methods.

### Option A: `kaggle.json`

Download your Kaggle API token from your Kaggle account settings, then run:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Option B: environment variables

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```



## One-command bootstrap

Run the full pipeline with:

```bash
bash scripts/bootstrap.sh
```

This performs the following steps:

1. downloads the M5 dataset
2. preprocesses raw files
3. builds forecasting features
4. trains the Seasonal Naive baseline
5. trains the LightGBM baseline
6. optionally trains the TFT model


## Fast start

To run the pipeline directly:

```bash
python -m src.run_pipeline --config configs/default.yaml
```

The default configuration is designed to be **laptop-friendly** and can automatically work on a reduced subset for a faster first pass.



## Train only selected models

Train only the baseline models:

```bash
python -m src.run_pipeline --config configs/default.yaml --models naive lightgbm
```

Train only TFT:

```bash
python -m src.run_pipeline --config configs/default.yaml --models tft
```



## Modeling stack

### 1. Seasonal Naive
A strong and necessary classical forecasting baseline.

Useful for checking whether more complex models actually beat a simple seasonal repeat strategy.

### 2. LightGBM
A tabular time-series baseline using engineered lag, rolling, calendar, and price features.

This often performs surprisingly well in industrial forecasting pipelines and is an important benchmark.

### 3. Temporal Fusion Transformer (TFT)
The main deep learning model in this repo.

TFT is well suited for structured multi-horizon forecasting because it supports:
- static covariates
- known future inputs
- observed historical inputs
- variable selection
- temporal attention
- interpretable components



## Evaluation

This repo uses a **time-based validation split**, not a random split.

Metrics include:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **WAPE** — Weighted Absolute Percentage Error
- **sMAPE** — Symmetric Mean Absolute Percentage Error





<div align="center">



</div>
