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
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "stock-forecasting-secret-key-2025")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        company1 = request.form["company1"].strip().upper()
        company2 = request.form["company2"].strip().upper()
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Convert input strings to datetime objects
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return render_template("index.html", error_message="Error: Date format is incorrect. Please use YYYY-MM-DD.")

        # Validate date range
        if start_date >= end_date:
            return render_template("index.html", error_message="Error: Start date must be before end date.")

        try:
            # Download historical data for the given companies
            app.logger.info(f"Fetching data for {company1} and {company2} from {start_date} to {end_date}")
            data1 = yf.download(company1, start=start_date, end=end_date, progress=False)
            data2 = yf.download(company2, start=start_date, end=end_date, progress=False)

            if data1 is None or data2 is None or data1.empty or data2.empty:
                return render_template("index.html", error_message="No data available for one or both symbols in the specified date range. Please check the stock symbols and date range.")

            # Ensure we have enough data for ARIMA modeling
            if len(data1) < 10 or len(data2) < 10:
                return render_template("index.html", error_message="Insufficient data for forecasting. Please select a longer date range (at least 10 trading days).")

            # Calculate the percentage difference in the last closing price between the two companies
            last_price1 = data1['Close'].iloc[-1]
            last_price2 = data2['Close'].iloc[-1]
            price_diff_pct = ((last_price2 - last_price1) / last_price1) * 100

            # Get current share prices for both companies
            try:
                company1_info = yf.Ticker(company1).history(period="1d")
                company2_info = yf.Ticker(company2).history(period="1d")
                current_price1 = company1_info['Close'].iloc[-1]
                current_price2 = company2_info['Close'].iloc[-1]
            except Exception as e:
                app.logger.error(f"Error fetching current prices: {e}")
                current_price1 = last_price1
                current_price2 = last_price2

            # Suggest which company is the best based on the most recent price
            best_company = company1 if current_price1 > current_price2 else company2

            # ARIMA Model Forecasting for Company 1
            try:
                df1 = data1[['Close']].dropna()
                train_size1 = max(5, int(len(df1) * 0.8))  # Ensure minimum training size
                train1, test1 = df1.iloc[:train_size1], df1.iloc[train_size1:]
                
                if len(test1) == 0:
                    # If no test data, use last 20% as forecast period
                    forecast_steps1 = max(1, int(len(df1) * 0.2))
                    train1 = df1
                    test1_index = pd.date_range(start=df1.index[-1], periods=forecast_steps1 + 1, freq='D')[1:]
                else:
                    test1_index = test1.index
                    forecast_steps1 = len(test1)

                model1 = ARIMA(train1['Close'], order=(1, 1, 2))
                model_fit1 = model1.fit()
                forecast1 = model_fit1.forecast(steps=forecast_steps1)

                # ARIMA Model Forecasting for Company 2
                df2 = data2[['Close']].dropna()
                train_size2 = max(5, int(len(df2) * 0.8))
                train2, test2 = df2.iloc[:train_size2], df2.iloc[train_size2:]
                
                if len(test2) == 0:
                    forecast_steps2 = max(1, int(len(df2) * 0.2))
                    train2 = df2
                    test2_index = pd.date_range(start=df2.index[-1], periods=forecast_steps2 + 1, freq='D')[1:]
                else:
                    test2_index = test2.index
                    forecast_steps2 = len(test2)

                model2 = ARIMA(train2['Close'], order=(1, 1, 2))
                model_fit2 = model2.fit()
                forecast2 = model_fit2.forecast(steps=forecast_steps2)

                # Plotting combined forecast for both companies
                plt.style.use('dark_background')
                plt.figure(figsize=(14, 8))
                
                plt.plot(train1.index, train1['Close'], label=f'{company1} Train', color='#203147', linewidth=2)
                if len(test1) > 0:
                    plt.plot(test1.index, test1['Close'], label=f'{company1} Test', color='#01ef63', linewidth=2)
                plt.plot(test1_index, forecast1, label=f'{company1} Forecast', color='orange', linewidth=2, linestyle='--')

                plt.plot(train2.index, train2['Close'], label=f'{company2} Train', color='#0044FF', linewidth=2)
                if len(test2) > 0:
                    plt.plot(test2.index, test2['Close'], label=f'{company2} Test', color='#FF6600', linewidth=2)
                plt.plot(test2_index, forecast2, label=f'{company2} Forecast', color='purple', linewidth=2, linestyle='--')

                plt.title(f'{company1} vs {company2} Close Price Forecast Comparison', fontsize=16, color='white')
                plt.xlabel('Date', fontsize=12, color='white')
                plt.ylabel('Close Price ($)', fontsize=12, color='white')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save combined forecast plot to a string
                img_combined = io.BytesIO()
                plt.savefig(img_combined, format='png', facecolor='#0f0f0f', edgecolor='none', dpi=100)
                img_combined.seek(0)
                combined_forecast_plot_url = base64.b64encode(img_combined.getvalue()).decode('utf8')
                plt.close()

                # Plotting forecast for Company 1
                plt.figure(figsize=(14, 8))
                plt.plot(train1.index, train1['Close'], label=f'{company1} Train', color='#203147', linewidth=2)
                if len(test1) > 0:
                    plt.plot(test1.index, test1['Close'], label=f'{company1} Test', color='#01ef63', linewidth=2)
                plt.plot(test1_index, forecast1, label=f'{company1} Forecast', color='orange', linewidth=2, linestyle='--')
                plt.title(f'{company1} Close Price Forecast', fontsize=16, color='white')
                plt.xlabel('Date', fontsize=12, color='white')
                plt.ylabel('Close Price ($)', fontsize=12, color='white')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save Company 1 forecast plot to a string
                img1 = io.BytesIO()
                plt.savefig(img1, format='png', facecolor='#0f0f0f', edgecolor='none', dpi=100)
                img1.seek(0)
                company1_forecast_plot_url = base64.b64encode(img1.getvalue()).decode('utf8')
                plt.close()

                # Plotting forecast for Company 2
                plt.figure(figsize=(14, 8))
                plt.plot(train2.index, train2['Close'], label=f'{company2} Train', color='#0044FF', linewidth=2)
                if len(test2) > 0:
                    plt.plot(test2.index, test2['Close'], label=f'{company2} Test', color='#FF6600', linewidth=2)
                plt.plot(test2_index, forecast2, label=f'{company2} Forecast', color='purple', linewidth=2, linestyle='--')
                plt.title(f'{company2} Close Price Forecast', fontsize=16, color='white')
                plt.xlabel('Date', fontsize=12, color='white')
                plt.ylabel('Close Price ($)', fontsize=12, color='white')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save Company 2 forecast plot to a string
                img2 = io.BytesIO()
                plt.savefig(img2, format='png', facecolor='#0f0f0f', edgecolor='none', dpi=100)
                img2.seek(0)
                company2_forecast_plot_url = base64.b64encode(img2.getvalue()).decode('utf8')
                plt.close()

                # Return the plots, price difference, and the best company suggestion
                return render_template(
                    "index.html",
                    combined_forecast_plot_url=combined_forecast_plot_url,
                    company1_forecast_plot_url=company1_forecast_plot_url,
                    company2_forecast_plot_url=company2_forecast_plot_url,
                    price_diff_pct=round(price_diff_pct, 2),
                    company1=company1,
                    company2=company2,
                    current_price1=round(current_price1, 2),
                    current_price2=round(current_price2, 2),
                    best_company=best_company
                )

            except Exception as e:
                app.logger.error(f"Error in ARIMA modeling: {e}")
                return render_template("index.html", error_message=f"Error in forecasting model: {str(e)}. Please try with different stocks or date range.")

        except Exception as e:
            app.logger.error(f"Error fetching stock data: {e}")
            return render_template("index.html", error_message=f"Error fetching stock data: {str(e)}. Please check the stock symbols and try again.")

    return render_template("index.html", combined_forecast_plot_url=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
