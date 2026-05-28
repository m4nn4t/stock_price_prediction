# Stock Price Prediction using GRU + EMA + RSI + CUDA

An advanced deep learning project for stock price forecasting using:

- GRU (Gated Recurrent Unit)
- Technical Indicators (EMA + RSI)
- CUDA GPU Acceleration
- TensorFlow / Keras
- Multi-feature Time Series Forecasting

This project predicts future stock prices while reducing prediction lag around peaks and dips using momentum-based indicators and a faster recurrent architecture.

---

#  Features

✅ CUDA GPU Acceleration  
✅ GRU-based Forecasting  
✅ EMA Technical Indicators  
✅ RSI Momentum Indicator  
✅ Multi-feature Learning  
✅ Reduced Prediction Lag  
✅ Parallel TensorFlow Data Pipeline  
✅ Google Colab Ready  
✅ Training / Testing Visualization  
✅ Multi-step Forecasting  

---

#  Project Goal

Traditional LSTM stock prediction models often:

- lag behind peaks
- smooth sharp movements
- react slowly to trend reversals

This project improves forecasting by combining:

```text
GRU
+
EMA
+
RSI
+
OHLCV Features
```

to better capture:

- momentum shifts
- volatility
- trend exhaustion
- turning points

---

# Why GRU Instead of LSTM?

GRU (Gated Recurrent Unit) is a simplified version of LSTM.

Advantages:

✅ Faster training  
✅ Fewer parameters  
✅ Better short-term adaptation  
✅ Reduced prediction lag  

---

#  Technical Indicators Used

---

## 1. EMA (Exponential Moving Average)

EMA gives more importance to recent prices.

### EMA 10

Captures short-term trend.

### EMA 30

Captures long-term trend.

### Why EMA helps

EMA reacts faster than normal moving averages and helps the model detect:

- momentum acceleration
- bullish/bearish crossover
- trend shifts

---

## 2. RSI (Relative Strength Index)

RSI measures market momentum.

Range:

```text
0 → 100
```

Interpretation:

| RSI | Meaning |
|---|---|
| >70 | Overbought |
| <30 | Oversold |

### Why RSI helps

RSI helps the model anticipate:

- reversals
- exhaustion
- market overheating

before price actually changes.

---

#  Architecture

```text
Input Features
(Open, High, Low, Close, Volume, EMA10, EMA30, RSI)

↓

GRU Layer (128)

↓

Dropout (0.05)

↓

GRU Layer (64)

↓

Dense Layer (64)

↓

Forecast Output
```

---

#  Features Used

```text
[
open,
high,
low,
close,
volume,
EMA_10,
EMA_30,
RSI
]
```

Total Features:

```text
8
```

---

# 📈 Forecast Strategy

Sliding Window:

```text
100 Previous Days
```

Forecast Horizon:

```text
3 Future Days
```

Transformation:

```text
[Day1 … Day100]

↓

[Day101 … Day103]
```

---

#  CUDA Optimization

This project uses GPU acceleration through CUDA.

Optimizations:

✅ TensorFlow GPU  
✅ Parallel Batch Processing  
✅ TensorFlow Prefetch Pipeline  
✅ XLA Kernel Fusion  

---

#  Parallelization Strategy

Without CUDA:

```text
CPU:
Batch1
↓

Batch2
↓

Batch3
```

With CUDA:

```text
GPU:
Batch1 Batch2 Batch3
↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
Parallel Matrix Operations
```

---

#  Dataset

Source:

```text
Tiingo Stock API
```

Ticker:

```text
AAPL
```

Period:

```text
2020 → 2025
```

---

#  Visualizations

---

## Forecast Plot

Save image:

```text
images/forecast.png
```

Add:

```md
## Forecast

![Forecast](images/forecast.png)
```

---

## Training Plot

Save image:

```text
images/train.png
```

Add:

```md
## Training Prediction

![Training](images/train.png)
```

---

## Testing Plot

Save image:

```text
images/test.png
```

Add:

```md
## Testing Prediction

![Testing](images/test.png)
```

---

## Combined Plot

Save image:

```text
images/full_plot.png
```

Add:

```md
## Combined Visualization

![Combined](images/full_plot.png)
```

---

#  Project Structure

```text
project/

│
├── notebook.ipynb
├── README.md
├── gru_stock_model.keras
├── gru_scaler.pkl
│
├── images/
│   ├── forecast.png
│   ├── train.png
│   ├── test.png
│   └── full_plot.png
```

---

# 🛠️ Installation

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install requests
pip install joblib
```

---

#  Run on Google Colab

Enable GPU:

```text
Runtime
↓

Change Runtime Type
↓

GPU
```

Then run notebook cells sequentially.

---

# Loss Function

```python
loss = "mse"
```

Why MSE?

- better peak sensitivity
- stronger penalty for large errors
- sharper curve fitting

---

# Results

The upgraded GRU + EMA + RSI model showed:

✅ Reduced prediction lag  
✅ Better peak tracking  
✅ Better volatility following  
✅ Improved test generalization  
✅ Faster GPU training  

---

# Limitations

Stock markets are highly stochastic.

Even advanced models cannot perfectly predict:

- sudden crashes
- news-driven spikes
- black swan events

Goal:

```text
reduce lag
minimize forecast error
capture momentum
```

not achieve perfect prediction.

---

# Future Improvements

- Attention Mechanism
- Transformer Forecasting
- Sentiment Analysis
- Candlestick Pattern Learning
- Hyperparameter Optimization
- Multi-stock Training
- Reinforcement Learning Trading Agent

---

# Author

Stock Price Forecasting using:

GRU + EMA + RSI + CUDA + TensorFlow
