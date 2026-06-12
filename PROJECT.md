# Project: Cross-Sectional Alpha Research System

## Objective

Build a modular alpha research system that generates, tests, and evaluates cross-sectional equity factors using daily market data from 2020-01-02 onward.

The goal is to discover statistically robust signals that improve long-short portfolio performance under realistic trading constraints.

---

## Data Schema

Data is stored as a nested JSON structure:

### Level 1: Date (str)

Example: "2023-01-05"

### Level 2: Stock ID

Example: "600000.SH"

### Level 3: Market Features

Each stock contains the following fields:

* open (float): opening price
* close (float): closing price
* high (float): highest price
* low (float): lowest price
* limit_up (float): upper limit price
* limit_down (float): lower limit price
* total_turnover (float): total traded value
* volume (float): trading volume
* num_trades (int): number of trades
* prev_close (float): previous day close (non-adjusted)

---

## Data Source (CRITICAL)

All market data is stored locally on the WSL filesystem.

### Data Directory

```text
/home/yifei_li/AlphaGPT/data/raw/
```

### Dataset Naming Convention

Market datasets follow the naming pattern:

```text
全A_行情数据_*.json
```

Examples:

```text
全A_行情数据_20_2306D.json
全A_行情数据_20_2406D.json
全A_行情数据_20_2506D.json
全A_行情数据_20_2606D.json
```

### Primary Dataset Selection Rule

Unless otherwise specified, agents should use the latest available dataset in the directory.

Current primary dataset:

```text
/home/yifei_li/AlphaGPT/data/raw/全A_行情数据_20_2606D.json
```

### Data Properties

* Format: JSON (nested dictionary)
* Frequency: Daily
* Universe: China A-share equities
* Coverage: 2020-01-02 onward
* Trading-calendar aligned

---

### Data Access Contract (MANDATORY)

All agents (ARCH, BOB, RICHARD, Builder, Reviewer) MUST treat the datasets in `/data/raw/` as the single source of truth.

Rules:

1. No external APIs
2. No third-party market datasets
3. No synthetic or simulated market data
4. All feature engineering must derive from the provided JSON files
5. Missing observations should be treated as NaN unless explicitly specified otherwise
6. Raw files must remain immutable

### Reference Loader

```python
from pathlib import Path
import json

DATA_DIR = Path("/home/yifei_li/AlphaGPT/data/raw")

latest_file = sorted(DATA_DIR.glob("全A_行情数据_*.json"))[-1]

with open(latest_file, "r") as f:
    market_data = json.load(f)
```

---

## Alpha Design Principles

All alpha factors must satisfy:

1. **Cross-sectional nature**

   * Computed across stocks at each timestamp

2. **No lookahead bias**

   * Only use information available up to time t

3. **Vectorizable computation**

   * Must be implementable using pandas / numpy / polars

4. **Market realism**

   * Respect limit_up / limit_down constraints where relevant

---

## Alpha Categories

### Category A — Price Momentum Structure

Capture short-term continuation and reversal effects.

* Alpha A1: 5-day return momentum
* Alpha A2: volatility-adjusted momentum (return / std)

---

### Category B — Liquidity & Trading Activity

Capture order flow pressure and market participation.

* Alpha B1: volume shock (volume vs rolling mean)
* Alpha B2: turnover intensity (turnover / market activity proxy)

---

### Category C — Price Efficiency & Microstructure

Capture inefficiencies and intraday structure.

* Alpha C1: high-low range expansion signal
* Alpha C2: close location value (CLV: where close sits in daily range)

---

## Evaluation Framework

Each alpha must be evaluated using:

### 1. Predictive Power

* Information Coefficient (IC)
* Rank IC

### 2. Portfolio Performance

* Long-short Sharpe ratio
* Max drawdown

### 3. Stability

* IC decay over time
* Turnover consistency

---

## Output Requirements (for Bob)

For each alpha:

1. Mathematical formula definition
2. Python implementation (vectorized)
3. Backtest function
4. Edge hypothesis explanation
5. Known failure cases

---

## System Constraints

* No external data allowed beyond provided JSON
* All computation must be reproducible
* Avoid survivorship bias assumptions
* Ensure alignment with trading calendar
