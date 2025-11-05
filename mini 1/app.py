import os
import logging
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "stock-forecasting-secret-key-2025")

def fetch_stock_data(company_symbol, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        data = yf.download(company_symbol, start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            return None, f"No data available for {company_symbol} in the specified date range."
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {company_symbol}: {e}"

def get_arima_forecast(data, company_symbol):
    """Performs ARIMA forecasting on the given stock data."""
    try:
        df = data[['Close']].dropna()
        if len(df) < 20:
            return None, None, None, None, "Insufficient data for forecasting. Please select a longer date range (at least 20 trading days)."

        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        
        model = ARIMA(train['Close'], order=(1, 1, 2))
        model_fit = model.fit()
        
        forecast_steps = len(test) if len(test) > 0 else int(len(df) * 0.2)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        test_index = test.index if len(test) > 0 else pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        
        return train, test, forecast, test_index, None
    except Exception as e:
        return None, None, None, None, f"Error in ARIMA modeling for {company_symbol}: {e}"

def create_plot(train, test, forecast, test_index, company_symbol, color_train, color_test, color_forecast):
    """Creates a plot for the given forecast data."""
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    plt.plot(train.index, train['Close'], label=f'{company_symbol} Train', color=color_train, linewidth=2)
    if len(test) > 0:
        plt.plot(test.index, test['Close'], label=f'{company_symbol} Test', color=color_test, linewidth=2)
    plt.plot(test_index, forecast, label=f'{company_symbol} Forecast', color=color_forecast, linewidth=2, linestyle='--')
    plt.title(f'{company_symbol} Close Price Forecast', fontsize=16, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Close Price ($)', fontsize=12, color='white')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='#0f0f0f', edgecolor='none', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        company1 = request.form["company1"].strip().upper()
        company2 = request.form["company2"].strip().upper()
        start_date_str = request.form["start_date"]
        end_date_str = request.form["end_date"]

        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            return render_template("index.html", error_message="Error: Date format is incorrect. Please use YYYY-MM-DD.")

        if start_date >= end_date:
            return render_template("index.html", error_message="Error: Start date must be before end date.")

        data1, error1 = fetch_stock_data(company1, start_date, end_date)
        if error1:
            return render_template("index.html", error_message=error1)
            
        data2, error2 = fetch_stock_data(company2, start_date, end_date)
        if error2:
            return render_template("index.html", error_message=error2)

        train1, test1, forecast1, test_index1, error1 = get_arima_forecast(data1, company1)
        if error1:
            return render_template("index.html", error_message=error1)

        train2, test2, forecast2, test_index2, error2 = get_arima_forecast(data2, company2)
        if error2:
            return render_template("index.html", error_message=error2)

        last_price1 = data1['Close'].iloc[-1]
        last_price2 = data2['Close'].iloc[-1]
        current_price1 = yf.Ticker(company1).history(period="1d")['Close'].iloc[-1]
        current_price2 = yf.Ticker(company2).history(period="1d")['Close'].iloc[-1]

        # Improved "best company" logic
        forecast_mean1 = np.mean(forecast1)
        forecast_mean2 = np.mean(forecast2)
        best_company = company1 if forecast_mean1 > forecast_mean2 else company2

        company1_plot_url = create_plot(train1, test1, forecast1, test_index1, company1, '#203147', '#01ef63', 'orange')
        company2_plot_url = create_plot(train2, test2, forecast2, test_index2, company2, '#0044FF', '#FF6600', 'purple')

        # Combined plot
        plt.figure(figsize=(14, 8))
        plt.style.use('dark_background')
        plt.plot(train1.index, train1['Close'], label=f'{company1} Train', color='#203147', linewidth=2)
        if len(test1) > 0:
            plt.plot(test1.index, test1['Close'], label=f'{company1} Test', color='#01ef63', linewidth=2)
        plt.plot(test_index1, forecast1, label=f'{company1} Forecast', color='orange', linewidth=2, linestyle='--')
        plt.plot(train2.index, train2['Close'], label=f'{company2} Train', color='#0044FF', linewidth=2)
        if len(test2) > 0:
            plt.plot(test2.index, test2['Close'], label=f'{company2} Test', color='#FF6600', linewidth=2)
        plt.plot(test_index2, forecast2, label=f'{company2} Forecast', color='purple', linewidth=2, linestyle='--')
        plt.title(f'{company1} vs {company2} Close Price Forecast Comparison', fontsize=16, color='white')
        plt.xlabel('Date', fontsize=12, color='white')
        plt.ylabel('Close Price ($)', fontsize=12, color='white')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        img_combined = io.BytesIO()
        plt.savefig(img_combined, format='png', facecolor='#0f0f0f', edgecolor='none', dpi=100)
        img_combined.seek(0)
        combined_plot_url = base64.b64encode(img_combined.getvalue()).decode('utf8')
        plt.close()

        return render_template(
            "index.html",
            combined_forecast_plot_url=combined_plot_url,
            company1_forecast_plot_url=company1_plot_url,
            company2_forecast_plot_url=company2_plot_url,
            company1=company1,
            company2=company2,
            current_price1=round(current_price1, 2),
            current_price2=round(current_price2, 2),
            best_company=best_company
        )

    return render_template("index.html", combined_forecast_plot_url=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
