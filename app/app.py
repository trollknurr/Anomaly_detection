from flask import Flask, request, jsonify, redirect, flash, url_for
import os
from influxdb_client import InfluxDBClient, Point
from datetime import datetime
from anomaly import get_anomaly
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'processed_data'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

client = InfluxDBClient(
    url="http://192.168.1.144:8086",
    token="Z0qLr5Uu_PfxhEs69u6tRIjn5A2KWAb12O-UXysIgUOA9hi1byHIaSnaXKb-e5mLUMsWPjqjRgO0KC3ecdp49w==",
    org="techlab"
)

write_api = client.write_api()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def get_anomaly_score():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.wav')
        #file.save(filepath)
    status, score = get_anomaly(request.files['file'])
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    write_api.write(
        "techlab",
        "techlab",
        Point("engine_break").field("proba", score)
    )
    return jsonify({
                    'datetime': dt_string,
                    'status': status,
                    'score': score})
    # else:
    #     return jsonify({
    #                     'error': 'hz'})

if __name__ == '__main__':
    #host=os.environ.get('HOST')
    #port = os.environ.get('PORT')
    app.run(debug=True, host='0.0.0.0', port=5001)
