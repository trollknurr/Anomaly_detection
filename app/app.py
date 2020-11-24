from flask import Flask, request, jsonify, redirect, flash, url_for
import os
from anomaly import get_anomaly
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'processed_data'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    status, score = get_anomaly(filepath)
    
    return jsonify({
                        'filename': filename,
                        'status': status, 
                        'score': score})

if __name__ == '__main__':
    host=os.environ.get('HOST')
    port = os.environ.get('PORT')
    app.run(debug=True, host=host, port=port)