# KNN-Implementation-in-Stock-Price-Prediction
A complete end-to-end implementation of K-Nearest Neighbors (KNN) for stock price prediction. Includes data preprocessing, feature engineering, model training, evaluation using MAE/MSE/RMSE, and visual comparison of predicted vs. actual stock trends. Ideal for learning regression techniques and building data science portfolio projects.
KNN Implementation in Stock Price Prediction

Overview
This project demonstrates how the K-Nearest Neighbors (KNN) algorithm can be applied to historical stock market data to predict future stock prices.
The workflow covers end-to-end steps including data sourcing, preprocessing, feature engineering, model training, evaluation, and visualization.

This project is suitable for:
Data Science freshers building a strong portfolio
Candidates showcasing ML fundamentals applied to real-world financial data
Anyone learning KNN, time-series preprocessing, and regression modelling

Key Features:

End-to-end machine learning pipeline for stock price prediction
Uses KNN Regression to forecast closing prices
Includes data cleaning, scaling, feature creation, and training/testing split
Error metrics such as MAE, MSE, RMSE for model evaluation
Visual comparisons of predicted vs. actual stock trends
Clean, modular notebook suitable for learning and interviews

Project Workflow
1. Data Collection:
Downloaded historical stock prices using libraries such as yfinance or existing CSV.
Columns used typically include:
Date
Open, High, Low, Close
Volume
2. Data Preprocessing
Handling missing values
Feature engineering (lag features, moving averages if applied)
Splitting into train and test sets
Applying MinMaxScaler for normalization (KNN is distance-based, so scaling is critical)

3. Model Development
Used KNN Regressor from sklearn
Hyperparameters tuned:
n_neighbors
distance metric
Model trained on historical windowed features to predict next-day close price

4. Model Evaluation
Metrics included:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Visualizations:
Line plot comparing predicted vs. actual values
Error distribution plots (if implemented)

Technologies Used


Python
NumPy, Pandas
Scikit-learn
Matplotlib / Seaborn
Jupyter Notebook
yFinance (optional)

Results Summary

Demonstrated accuracy of KNN in capturing local patterns in stock price movements
Showed limitations of non-sequential ML models in long-term forecasting
Provided a strong baseline that can be improved with:
LSTM/GRU deep learning models
Feature engineering (RSI, MACD, Moving Averages)
Cross-validation tuning

How to Run the Project
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git

2. Install dependencies
pip install -r requirements.txt

3. Open the notebook
jupyter notebook knn_implementation_in_stock.ipynb

4. Execute all cells to reproduce results
Future Enhancements

Add LSTM/Transformer-based models for comparison

Automate daily stock data pulling

Deploy predictions using a Flask/FastAPI backend

Build dashboards with Streamlit

Project Motivation

This project helps demonstrate:

Practical application of ML algorithms

Real-world data processing skills

Understanding of regression, evaluation metrics, and modelling limitations
It is ideal for showcasing competencies in Data Science and AI Engineering during interviews.
