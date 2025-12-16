# RNN LSTM Time Series Prediction 

A Python project using Recurrent Neural Networks (RNN) with LSTM layers to predict future values of a time series (stock prices, weather, etc.).

---

## Project Overview  
This project preprocesses time series data, builds an LSTM model, trains it, and visualizes predictions of the real dataset of Google stock prices (from Kaggle).

---
##  Mathematical Intuition (Stock Price Prediction using RNN / LSTM)

###  Why Time Series Models Are Needed
Stock prices are **sequential data**, meaning:
- Today’s price depends on past prices
- Patterns unfold over time, not independently

Traditional machine learning models treat each data point independently, which fails to capture **temporal dependencies**.  
Recurrent Neural Networks (RNNs) are designed to model such sequences.

---

###  Recurrent Neural Network (RNN) Intuition
An RNN processes data **one timestep at a time** while maintaining a hidden state that acts as memory.

At each timestep:
- The current input (price or feature)
- The previous hidden state

are combined to produce:
- A new hidden state
- An output prediction

This allows the model to learn **short-term temporal patterns** in stock prices.

However, standard RNNs struggle with **long-term dependencies** due to vanishing gradients.

---

###  Why LSTM Works Better
Long Short-Term Memory (LSTM) networks solve this problem by introducing **gated memory cells**.

Each LSTM cell contains:
- A **forget gate** that decides what past information to discard
- An **input gate** that decides what new information to store
- An **output gate** that controls what information influences the prediction

This gating mechanism allows LSTMs to:
- Preserve important trends over long periods
- Ignore short-term noise
- Learn smoother and more stable temporal patterns

---

###  Learning Objective
The model is trained to predict the **next price (or return)** based on a window of previous prices.

Training minimizes **Mean Squared Error (MSE)**:
- Large errors are penalized more heavily
- Encourages predictions to stay close to actual prices

As a result, predictions tend to be **smoothed versions of real price movements**.

---

###  Why Predictions Look Smooth
Stock markets are noisy and influenced by external events.

Since the model:
- Learns average historical patterns
- Optimizes for squared error

It tends to:
- Smooth sudden spikes
- Lag during abrupt market changes

This is expected behavior and reflects the **limitations of purely historical models**.

---

###  Key Limitation (Important)
The model assumes:
- Past price patterns contain predictive information
- Market behavior is somewhat stationary

It cannot:
- Predict sudden news-driven events
- Model regime shifts without additional features

---

##  Features  
- Time series data preprocessing (normalization, windowing)  
- LSTM model implementation in TensorFlow / PyTorch  
- Model training, evaluation, and saving  
- Prediction vs. True Value plots

---

##  Tech Stack  
- Python 3.10  
- TensorFlow / PyTorch  
- Pandas, NumPy  
- Matplotlib, Jupyter Notebook

---

##  Installation & Usage  

1. Clone this repo:  
   ```bash
   git clone https://github.com/harmansingh/rnn-lstm-timeseries.git


   pip install -r requirements.txt

