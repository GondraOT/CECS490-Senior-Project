# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# HoopIQ Cloud API
# Flask backend for receiving data from Raspberry Pi and serving to website

from flask import Flask, render_template
from flask_cors import CORS
from app import app

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # <-- processes Jinja2 includes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
