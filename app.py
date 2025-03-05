import requests
import pandas as pd
from datetime import datetime
import io
import random
from prophet import Prophet
import os
import time
from functools import lru_cache
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

portfolio = {}
user_scores = {}  # For gamification
eco_scores = {
    "AAPL": {"score": 75, "carbon": 4500},
    "MSFT": {"score": 80, "carbon": 3800},
    "TSLA": {"score": 95, "carbon": 2000}
}

@lru_cache(maxsize=128)  # Cache results to avoid repeated API calls
def get_stock_data(ticker):
    try:
        # Add a 1-second delay to respect API rate limits
        time.sleep(1)
        logger.info(f"Fetching data for ticker: {ticker}")
        api_key = "your_fmp_api_key"  # Replace with your FMP API key from financialmodelingprep.com
        base_url = "https://financialmodelingprep.com/api/v3"

        # Get quote (real-time price)
        quote_url = f"{base_url}/quote/{ticker}?apikey={api_key}"
        quote_response = requests.get(quote_url, timeout=10)
        quote_response.raise_for_status()
        quote_data = quote_response.json()[0]
        current_price = quote_data.get('price', 0)

        # Get historical data (30 days)
        history_url = f"{base_url}/historical-price-full/{ticker}?from={datetime.now().date() - pd.Timedelta(days=30)}&to={datetime.now().date()}&apikey={api_key}"
        history_response = requests.get(history_url, timeout=10)
        history_response.raise_for_status()
        history_data = history_response.json()['historical']
        history_df = pd.DataFrame(history_data)
        if history_df.empty or 'close' not in history_df:
            raise ValueError(f"No historical data available for ticker: {ticker}")
        
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date')
        close_prices = history_df['close'].tail(30).tolist()
        sma_20 = history_df['close'].rolling(window=20).mean().iloc[-1]
        rsi = calculate_rsi(history_df['close'].tail(14).tolist())  # Simplified RSI calculation

        # AI Prediction (simulated with Prophet)
        df = pd.DataFrame({"ds": history_df['date'], "y": history_df["close"]})
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=7)  # 7-day forecast
        forecast = model.predict(future)
        prediction = round(forecast["yhat"].iloc[-1], 2)

        logger.info(f"Successfully fetched data for ticker: {ticker}")
        return {
            "name": quote_data.get('name', ticker),
            "price": round(current_price, 2),
            "sma_20": round(sma_20, 2) if not pd.isna(sma_20) else 0,
            "rsi": round(rsi, 2) if rsi is not None else 0,
            "decision": "Buy" if current_price < sma_20 else "Sell" if current_price > sma_20 else "Hold",
            "volume": quote_data.get('volume', 0),
            "change": round(((current_price - history_df['close'].iloc[-2]) / history_df['close'].iloc[-2]) * 100, 2) if len(history_df['close']) > 1 else 0,
            "chart_data": close_prices,
            "prediction": prediction,
            "eco_score": eco_scores.get(ticker, {"score": 50, "carbon": 5000})
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_rsi(prices):
    if len(prices) < 14:
        return None
    deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:14]) / 14
    avg_loss = sum(losses[:14]) / 14
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_stock_news(ticker):
    try:
        api_key = "your_fmp_api_key"  # Replace with your FMP API key
        news_url = f"https://financialmodelingprep.com/api/v4/stock_news?ticker={ticker}&limit=3&apikey={api_key}"
        response = requests.get(news_url, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        return [{"title": item['title'], "link": item['url']} for item in news_data]
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

def get_financial_tips(user_data=None):
    tips = [
        "Save 10% of your income monthly for a rainy day!",
        "Consider diversifying with international stocks.",
        "Check your portfolio’s eco-impact weekly."
    ]
    return random.choice(tips)

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    if request.method == "POST":
        ticker = request.form["ticker"].upper().strip()
        data = get_stock_data(ticker)
        if data:
            portfolio[ticker] = data
        else:
            error = f"Invalid ticker: {ticker}"
    
    news = {ticker: get_stock_news(ticker) for ticker in portfolio.keys() if get_stock_data(ticker)}
    ai_tip = get_financial_tips()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Embed index.html content as a string with Jinja2 variables
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kalilfin - Your Financial Edge</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.11.5/gsap.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f4f4f4; transition: background-color 0.3s, color 0.3s; }
            .dark-mode { background-color: #1a1a1a; color: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
            .dark-mode .container { background: #2a2a2a; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
            .header { text-align: center; padding: 20px 0; }
            .header h1 { color: #2a67b3; font-size: 2.5em; margin: 0; letter-spacing: 2px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
            .dark-mode .header h1 { color: #4a87d3; text-shadow: 1px 1px 2px rgba(255,255,255,0.1); }
            .header span { font-size: 1em; color: #666; font-style: italic; }
            .dark-mode .header span { color: #aaa; }
            form, .actions { display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; justify-content: center; }
            input[type="text"] { padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 1.1em; flex-grow: 1; }
            .dark-mode input[type="text"] { background: #333; color: #fff; border-color: #555; }
            button { padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 1.1em; transition: background-color 0.2s, transform 0.2s; }
            .add-btn { background-color: #2a67b3; color: white; }
            .buy-btn { background-color: #4CAF50; color: white; }
            .add-btn:hover, .buy-btn:hover { background-color: #245c9e; transform: scale(1.05); }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #fff; border-radius: 8px; overflow: hidden; }
            th, td { padding: 14px; text-align: left; border-bottom: 1px solid #ddd; }
            .dark-mode th, .dark-mode td { border-color: #555; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .dark-mode th { background-color: #3a3a3a; }
            .decision-buy { color: #4CAF50; font-weight: bold; }
            .decision-sell { color: #e63946; font-weight: bold; }
            .decision-hold { color: #666; }
            .error { color: #e63946; text-align: center; margin: 10px 0; font-size: 1.1em; }
            .timestamp { font-size: 1em; color: #666; text-align: center; margin: 10px 0; }
            .dark-mode .timestamp { color: #aaa; }
            .remove-btn, .chart-btn { color: #2a67b3; cursor: pointer; text-decoration: underline; transition: color 0.2s; }
            .dark-mode .remove-btn, .dark-mode .chart-btn { color: #4a87d3; }
            .remove-btn:hover, .chart-btn:hover { color: #1e4b7a; }
            .stats { text-align: center; margin: 20px 0; font-size: 1.2em; color: #333; background: #f9f9f9; padding: 15px; border-radius: 8px; }
            .dark-mode .stats { color: #ddd; background: #333; }
            .news, .eco, .ai-coach, .challenges, .crypto-nft { margin: 20px 0; font-size: 1em; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
            .dark-mode .news, .dark-mode .eco, .dark-mode .ai-coach, .dark-mode .challenges, .dark-mode .crypto-nft { background: #2a2a2a; box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
            .news a { color: #2a67b3; text-decoration: none; }
            .dark-mode .news a { color: #4a87d3; }
            canvas { max-width: 100%; margin: 20px auto; display: block; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
            .actions { justify-content: center; }
            .export-btn, .dark-mode-toggle { padding: 12px 24px; background-color: #2a67b3; color: white; text-decoration: none; border-radius: 8px; margin: 5px; transition: background-color 0.2s, transform 0.2s; }
            .export-btn:hover, .dark-mode-toggle:hover { background-color: #245c9e; transform: scale(1.05); }
            .badge { display: inline-block; padding: 5px 10px; background: #ffd700; color: #2a67b3; border-radius: 4px; margin: 5px; animation: badgePop 0.5s ease-out; }
            @keyframes badgePop { 0% { transform: scale(0); opacity: 0; } 50% { transform: scale(1.2); opacity: 0.8; } 100% { transform: scale(1); opacity: 1; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Kalilfin</h1>
                <span>Your Financial Edge</span>
            </div>
            <div class="stats">
                <p>Portfolio Snapshot: Coming Soon!</p>
            </div>
            <form method="POST">
                <input type="text" name="ticker" placeholder="Enter stock ticker (e.g., AAPL)" required>
                <button type="submit" class="add-btn">Add Stock</button>
            </form>
            
            {% if error %}
                <p class="error">{{ error }}</p>
            {% endif %}
            
            {% if portfolio %}
                <p class="timestamp">Last updated: {{ timestamp }}</p>
                <table>
                    <tr>
                        <th>Company</th>
                        <th>Price ($)</th>
                        <th>Change (%)</th>
                        <th>SMA (20d)</th>
                        <th>RSI</th>
                        <th>Volume</th>
                        <th>Recommendation</th>
                        <th>Action</th>
                        <th>Prediction ($)</th>
                        <th>Eco Score</th>
                    </tr>
                    {% for ticker, data in portfolio.items() %}
                        <tr>
                            <td>{{ data.name }}</td>
                            <td>{{ data.price }}</td>
                            <td>{{ data.change }}</td>
                            <td>{{ data.sma_20 }}</td>
                            <td>{{ data.rsi }}</td>
                            <td>{{ data.volume }}</td>
                            <td class="decision-{{ data.decision.lower() }}">{{ data.decision }}</td>
                            <td>
                                <span class="remove-btn" onclick="removeStock('{{ ticker }}')">Remove</span> |
                                <span class="chart-btn" onclick="showChart('{{ ticker }}', {{ data.chart_data|tojson }})">Chart</span>
                            </td>
                            <td>{{ data.prediction }}</td>
                            <td>{{ data.eco_score.score }} ({{ data.eco_score.carbon }} kg CO2)</td>
                        </tr>
                    {% endfor %}
                </table>
                <canvas id="stockChart" style="display: none;"></canvas>
                <div class="news">
                    <h3>Latest News</h3>
                    {% for ticker, articles in news.items() %}
                        <p><strong>{{ portfolio[ticker].name }}</strong>:</p>
                        <ul>
                            {% for article in articles %}
                                <li><a href="{{ article.link }}" target="_blank">{{ article.title }}</a></li>
                            {% endfor %}
                        </ul>
                    {% endfor %}
                </div>
                <div class="eco">
                    <h3>Green Investing</h3>
                    <p>Explore our eco-friendly investments! Top Green Pick: TSLA (Eco Score: 95, Carbon: 2000 kg CO2).</p>
                    <canvas id="ecoChart" width="400" height="200"></canvas>
                    <script>
                        const ecoCtx = document.getElementById('ecoChart').getContext('2d');
                        new Chart(ecoCtx, {
                            type: 'bar',
                            data: {
                                labels: ['AAPL', 'MSFT', 'TSLA'],
                                datasets: [{
                                    label: 'Eco Score',
                                    data: [75, 80, 95],
                                    backgroundColor: '#4CAF50',
                                    borderColor: '#2a67b3',
                                    borderWidth: 1
                                }]
                            },
                            options: { scales: { y: { beginAtZero: true, max: 100 } } }
                        });
                    </script>
                </div>
                <div class="ai-coach">
                    <h3>Financial Wellness Coach</h3>
                    <p>{{ ai_tip }}</p>
                    <button onclick="playCoachTip()">Hear More Tips!</button>
                    <script>
                        function playCoachTip() {
                            const audio = new Audio('https://www.myinstants.com/media/sounds/sample-audio-file.mp3');
                            audio.play();
                        }
                    </script>
                </div>
                <div class="challenges">
                    <h3>Investment Challenges</h3>
                    <p>Beat the Market Challenge: Earn {{ random.randint(100, 1000) }} points this month!</p>
                    <p>Leaderboard: 1. Kalil Jamal - 500 pts <span class="badge">Stock Guru</span></p>
                </div>
                <div class="crypto-nft">
                    <h3>Crypto & NFT Hub</h3>
                    <p>Bitcoin: $50,000 (Bullish 85%) | Top NFT: CryptoPunk #1234 (Value: $100,000)</p>
                    <canvas id="cryptoChart" width="400" height="200"></canvas>
                    <script>
                        const cryptoCtx = document.getElementById('cryptoChart').getContext('2d');
                        new Chart(cryptoCtx, {
                            type: 'line',
                            data: {
                                labels: ['Day 1', 'Day 2', 'Day 3'],
                                datasets: [{
                                    label: 'Bitcoin Price',
                                    data: [48000, 50000, 52000],
                                    borderColor: '#ffd700',
                                    fill: false
                                }]
                            },
                            options: { scales: { y: { beginAtZero: false } } }
                        });
                    </script>
                </div>
                <div class="actions">
                    <a href="/export" class="export-btn">Export Portfolio</a>
                    <button class="dark-mode-toggle" onclick="toggleDarkMode()">Toggle Dark Mode</button>
                </div>
            {% else %}
                <p style="text-align: center;">Add a stock to start your journey with Kalilfin!</p>
            {% endif %}
        </div>

        <script>
            let chartInstance = null;

            function removeStock(ticker) {
                fetch(`/remove/${ticker}`).then(() => location.reload());
            }

            function showChart(ticker, data) {
                const ctx = document.getElementById('stockChart').getContext('2d');
                document.getElementById('stockChart').style.display = 'block';
                if (chartInstance) chartInstance.destroy();
                chartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: Array.from({ length: data.length }, (_, i) => i + 1),
                        datasets: [{
                            label: `${ticker} Price (30d)`,
                            data: data,
                            borderColor: '#2a67b3',
                            fill: false
                        }]
                    },
                    options: { scales: { y: { beginAtZero: false } } }
                });
            }

            function toggleDarkMode() {
                document.body.classList.toggle('dark-mode');
            }

            // Gamification Animation
            gsap.from(".badge", { duration: 0.5, scale: 0, opacity: 0, stagger: 0.2 });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content, portfolio=portfolio, news=news, error=error, timestamp=timestamp, ai_tip=ai_tip)

@app.route("/remove/<ticker>")
def remove_stock(ticker):
    portfolio.pop(ticker, None)
    return jsonify({"status": "success"})

@app.route("/export")
def export_portfolio():
    if not portfolio:
        return "Portfolio is empty!", 400
    df = pd.DataFrame(portfolio).T
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"kalilfin_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
