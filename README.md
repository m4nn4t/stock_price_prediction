# 📈 Stock Price Prediction using Stacked LSTM (TensorFlow)

A deep learning project that predicts future stock prices using a **Stacked LSTM (Long Short-Term Memory)** network trained on historical stock closing prices.

The model fetches historical stock data, preprocesses it using MinMax scaling, trains a multi-layer LSTM network, and recursively forecasts the next **30 trading days**.

---

## Project Overview

This project performs:

1. Historical stock data collection
2. Data preprocessing and normalization
3. Sliding-window sequence generation
4. Training a stacked LSTM model
5. Price prediction
6. Evaluation using RMSE
7. Future forecasting (next 30 days)
8. Model serialization

---

## Architecture

```text
Historical Stock Data
         ↓
Close Price Extraction
         ↓
MinMax Scaling
         ↓
Sliding Window Creation
(100 Previous Days)
         ↓
Stacked LSTM
(50 → 50 → 50)
         ↓
Dense Layer
         ↓
Next Day Prediction
         ↓
Recursive Forecasting
(30 Days)
```

---

## Dataset

Data Source:
- Tiingo Stock API

Ticker Used:
- AAPL (Apple)

Date Range:
- 2020-01-01 → 2025-11-01

Feature Used:
- Closing Price only

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Requests
- Tiingo API

---

# Model Architecture

### Layer 1

```python
LSTM(50, return_sequences=True)
```

Purpose:
- Learns short-term trends
- Extracts temporal features

Output:

```text
(100 × 50)
```

---

### Layer 2

```python
LSTM(50, return_sequences=True)
```

Purpose:
- Learns higher-level sequential patterns
- Builds deeper representation

Output:

```text
(100 × 50)
```

---

### Layer 3

```python
LSTM(50)
```

Purpose:
- Compresses entire sequence into a final representation

Output:

```text
(50)
```

---

### Output Layer

```python
Dense(1)
```

Purpose:
- Predict next stock closing price

Output:

```text
Single Value
```

---

# Sliding Window Mechanism

Window Size:

```text
100 Days
```

Example:

Input:

```text
[100,102,104,...100 days]
```

Target:

```text
Day 101 Price
```

Sliding:

```text
Window 1:
[1..100] → 101

Window 2:
[2..101] → 102

Window 3:
[3..102] → 103
```

---

## Training Configuration

```python
Epochs = 100
Batch Size = 64
Loss = Mean Squared Error
Optimizer = Adam
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/stock-price-lstm.git
cd stock-price-lstm
```

Install dependencies:

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install requests
pip install pandas_datareader
```

---

## Run Project

Execute:

```bash
python train_script.py
```

Outputs:

```text
AAPL.csv
lstm_stock_model.h5
scaler.pkl
```

---

## Prediction Workflow

```text
Last 100 Days
      ↓
Predict Day 1
      ↓
Append Prediction
      ↓
Predict Day 2
      ↓
Append Prediction
      ↓
...
      ↓
Predict Day 30
```

This is called:

```text
Recursive Forecasting
```

---

## Evaluation Metric

RMSE (Root Mean Squared Error)

Formula:

```text
RMSE = √(Σ(y−ŷ)² / N)
```

Lower RMSE indicates better predictive performance.

---

## Future Improvements

- CUDA GPU acceleration
- Bidirectional LSTM
- GRU comparison
- Transformer forecasting
- Multi-feature input:
  - Open
  - High
  - Low
  - Volume
- Direct 30-day prediction
- Hyperparameter tuning
- Deploy using FastAPI

---

## Limitations

- Uses only closing prices
- Recursive forecasting accumulates error
- No market sentiment
- No technical indicators
- Limited long-horizon accuracy

---

## Saved Artifacts

### Trained Model

```text
lstm_stock_model.h5
```

### Scaler

```text
scaler.pkl
```

---

## Example Output

```text
Input:
Previous 100 stock prices

Output:
Predicted next 30 days
```

---

## Author

Stock Price Prediction using Deep Learning and LSTM
Built with TensorFlow + Keras
