# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/routes/home.py

from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__)

@home_bp.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # your main page with Jinja includes