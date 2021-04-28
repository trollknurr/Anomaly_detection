import os
import tempfile

from flask import Flask, request, jsonify, redirect, flash, url_for, render_template
from influxdb_client import InfluxDBClient, Point
from anomaly import get_anomaly

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'someobviousshit'

client = InfluxDBClient(
    url="http://influxdb:8086",
    token="ixfIXVQmewr59n4SheTgWwQmWw659hlB3Qv8xnuvPSbLhHOU3JUk5OanbxBBmd9Ij26h-_wS5HuqBCHUZ4hd0g==",
    org="techlab"
)

write_api = client.write_api()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def get_anomaly_score():
    ogg_file = request.files['file']
    with tempfile.NamedTemporaryFile(suffix='.ogg') as tf:
        ogg_file.save(tf.name)
        status, score = get_anomaly(tf.name)

    write_api.write(
        "techlab",
        "techlab",
        Point("remote_mic").field("score", score)
    )
    return jsonify({'status': status, 'score': score})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000,  ssl_context=('cert.pem', 'key.pem'))
