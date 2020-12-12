from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import uuid
import os
from predict import Predict
# import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def save_file(f):
    filename = str(uuid.uuid4())+ '_' + f.filename  
    filename = secure_filename(filename)
    f.save(os.path.join(UPLOAD_FOLDER, filename))
    image_path = UPLOAD_FOLDER + '/' + filename
    return image_path

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['GET','POST'])
def result():
    if request.method == 'POST':
        image = request.files.get('myfile')
        image_path = save_file(image)
        image_name = './static/uploads/' + image_path.split('/')[-1]
        predictions = Predict(image_path)
        return render_template('result.html', predictions = predictions, image = image_name)

if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
