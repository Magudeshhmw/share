# Stock Forecasting Application

A Flask-based web application that provides advanced stock price forecasting using ARIMA (AutoRegressive Integrated Moving Average) models.

## Features

- Compare two stock symbols over a specified date range
- Generate predictive forecasts with ARIMA models
- Visual charts showing historical data and forecasts
- Investment recommendations based on current prices
- Responsive modern UI with dark theme

## Installation

1. Install Python 3.11 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:5000`

3. Enter two stock symbols (e.g., AAPL, MSFT)
4. Select date range
5. Click "Generate Forecast" to see the analysis

## Technology Stack

- **Backend**: Flask (Python)
- **Data Source**: Yahoo Finance API (yfinance)
- **Machine Learning**: ARIMA models (statsmodels)
- **Visualization**: Matplotlib
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript

## Deployment

For production deployment with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## File Structure

- `app.py` - Main Flask application
- `main.py` - Application entry point for WSGI
- `templates/index.html` - Frontend template
- `static/styles.css` - Styling and responsive design
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Modern Python project configuration

## License

Open source - feel free to modify and use as needed.#   s h a r e  
 #   s h a r e  
 