from flask import Flask, render_template, jsonify
import time
from metrics.Metric1.stock_picking import get_metric1_stocks
from metrics.Metric2.stock_picking import get_metric2_stocks
from metrics.Metric3.stock_picking import get_metric3_stocks
from threading import Thread
import pandas as pd

app = Flask(__name__)

# Global variable to store the latest data
latest_data = {
    'metric1': [],
    'metric2': [],
    'metric3': []
}

def update_metrics():
    while True:
        # Update Metric1
        latest_data['metric1'] = get_metric1_stocks()
        time.sleep(1)  # Small delay between updates
        
        # Update Metric2
        latest_data['metric2'] = get_metric2_stocks()
        time.sleep(1)
        
        # Update Metric3
        latest_data['metric3'] = get_metric3_stocks()
        
        # Wait for 30 minutes before next update
        time.sleep(500)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_metric1')
def get_metric1():
    return jsonify(latest_data['metric1'])

@app.route('/get_metric2')
def get_metric2():
    return jsonify(latest_data['metric2'])

@app.route('/get_metric3')
def get_metric3():
    return jsonify(latest_data['metric3'])

if __name__ == '__main__':
    # Start the background thread for updating metrics
    update_thread = Thread(target=update_metrics, daemon=True)
    update_thread.start()
    
    app.run(debug=True)