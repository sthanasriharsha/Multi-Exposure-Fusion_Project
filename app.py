
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from utils import *
from filters import *

app = Flask(__name__)

UPLOAD_FOLDER = 'imgs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def HDR(path, flag):
    image = cv2.imread(path)
    S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    S = S + 1e-20
    image = 1.0 * image / 255

    if flag:
        I = gdft(S, 3)
    else:
        I = wlsFilter(S)
    mI = np.mean(I)
    R = np.log(S + 1e-20) - np.log(I + 1e-20)
    R_eh = SRS(R, I)

    v_s = [0.2, (mI + 0.2) / 2, mI, (mI + 0.8) / 2, 0.8]

    I_vts = VIG(I, 1.0 - I, v_s)
    L_eh = tone_production(R_eh, I_vts)

    ratio = np.clip(L_eh / S, 0, 3)
    b, g, r = cv2.split(image)

    b_eh = ratio * b
    g_eh = ratio * g
    r_eh = ratio * r

    out = cv2.merge((b_eh, g_eh, r_eh))
    return np.clip(out, 0.0, 1.0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded image
            out = HDR(filepath, True)

            # Save the result
            original_path = os.path.join(app.config['STATIC_FOLDER'], filename)
            enhanced_name = 'rs_' + filename
            enhanced_path = os.path.join(app.config['STATIC_FOLDER'], enhanced_name)
            cv2.imwrite(enhanced_path, np.uint8(out * 255))

            return render_template('result.html', original_image=filename, enhanced_image=enhanced_name)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=5004)
