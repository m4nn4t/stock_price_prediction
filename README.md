# Optimized Stock Price Prediction using Multi-Step LSTM + TensorFlow

Predict future stock prices using an optimized **Multi-Step LSTM Neural Network**.

This project forecasts the **next 30 days of stock prices directly in one forward pass** instead of recursively predicting one day at a time.

Built using:

- TensorFlow / Keras
- LSTM
- Bidirectional LSTM
- GPU Acceleration (Google Colab CUDA)
- Tiingo Stock API
- MinMax Scaling
- Multi-Step Forecasting

---

# Project Overview

Traditional LSTM forecasting predicts:

```text
Day 1
↓

Day 2 using Day 1

↓

Day 3 using Day 2
```

This causes **error accumulation**.

This project improves the pipeline by predicting:

```text
Previous 100 Days
↓

Predict Entire Next 30 Days
```

Result:

✅ Faster inference  
✅ More stable predictions  
✅ Better GPU utilization  
✅ Reduced long-term drift  

---

# Model Architecture

```text
Input
(100 Days)

↓

Bidirectional LSTM
128 Units

↓

Dropout (0.2)

↓

LSTM
64 Units

↓

Dropout (0.2)

↓

Dense (64)

↓

Dense (30)

↓

Next 30 Day Prediction
```

---

# Dataset

Source:
Tiingo API

Ticker:
AAPL

Period:

```text
2020 → 2025
```

Input Feature:

```text
Close Price
```

---

# Sliding Window

Window Size:

```text
100 Days
```

Forecast Horizon:

```text
30 Days
```

Transformation:

```text
[Day1 … Day100]

↓

[Day101 … Day130]
```

---

# Optimizations Applied

## 1. Direct Multi-Step Forecasting

Old:

```text
100 → 1 → 1 → 1
```

New:

```text
100 → 30
```

Removes recursive prediction error.

---

## 2. Bidirectional LSTM

Learns patterns in both directions.

```text
Forward
↓

Backward
```

---

## 3. Dropout

Prevents overfitting.

```text
Dropout(0.2)
```

---

## 4. Early Stopping

Stops training automatically.

```text
patience=10
```

---

## 5. Adaptive Learning Rate

```text
ReduceLROnPlateau()
```

---

# Training Results

Training converged early using validation monitoring.

Example:

```text
loss ≈ 0.004

val_loss ≈ 0.00085
```

---

# Forecast Visualization

Save your image as:

```text
images/forecast.png
```

Then show it:

```md
## Next 30 Day Forecast

![Forecast](images/forecast.png)
```

Example:

## Next 30 Day Forecast

![Forecast](images/forecast.png)

---

# Training Performance

Save image as:

```text
images/train_vs_actual.png
```

Add:

```md
## Training Set Performance

![Training](images/train_vs_actual.png)
```

Example:

## Training Set Performance

![Training](images/train_vs_actual.png)

---

# Install

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

# Run

Google Colab:

```bash
Runtime
↓

Change Runtime

↓

GPU
```

Execute notebook cells.

---

# Outputs

```text
optimized_stock_model.keras

optimized_scaler.pkl

30 Day Forecast Graph

Train vs Actual Graph
```

---

# Project Structure

```text
project/

│
├── notebook.ipynb
├── optimized_stock_model.keras
├── optimized_scaler.pkl
├── README.md
│
├── images/
│   ├── forecast.png
│   └── train_vs_actual.png
```

---

# Future Improvements

- Transformer Forecasting
- Attention Mechanism
- Technical Indicators
- Sentiment Analysis
- Multi-Stock Training
- Hyperparameter Search

---

# Author

Stock Price Prediction using Optimized Multi-Step LSTM
TensorFlow + GPU + Google Colab
