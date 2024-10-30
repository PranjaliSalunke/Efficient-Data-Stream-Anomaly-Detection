import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, Response, render_template
import io

app = Flask(__name__)

latest_value = 0
values = []  # Store last 50 values for anomaly detection

def data_stream():
    global latest_value
    t = 0
    while True:
        seasonal_value = np.sin(t) * 10
        noise = np.random.normal(0, 2)
        latest_value = seasonal_value + noise
        yield latest_value
        t += 0.1

def detect_anomalies(values, threshold=2):
    if len(values) < 3:
        return []
    mean = np.mean(values)
    std_dev = np.std(values)
    anomalies = []

    for i, value in enumerate(values):
        z_score = (value - mean) / std_dev if std_dev > 0 else 0
        if abs(z_score) > threshold:
            anomalies.append(i)
    return anomalies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plot')
def plot():
    global values
    threshold = 100
    stream = data_stream()

    for _ in range(50):
        value = next(stream)
        values.append(value)
        if len(values) > 50:
            values.pop(0)

    plt.figure(figsize=(10, 5))
    plt.plot(values, label="Data Stream", color="#336699", linewidth=1.5)
    plt.axhline(y=threshold, color="#55aa55", linestyle="--", linewidth=1.2, label="Static Threshold")
    plt.axhline(y=-threshold, color="#55aa55", linestyle="--", linewidth=1.2)
    plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    anomalies = detect_anomalies(values)
    for i in anomalies:
        plt.plot(i, values[i], 'ro', markersize=6, label="Anomaly" if i == anomalies[0] else "")

    plt.legend(loc="upper right", fontsize=9)
    plt.title("Real-Time Data Stream with Anomaly Detection", fontsize=14, color="#333")
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    img.seek(0)

    return Response(img.getvalue(), mimetype='image/png')

@app.route('/latest_value')
def latest_value_route():
    return str(latest_value)

if __name__ == '__main__':
    app.run(debug=True)
