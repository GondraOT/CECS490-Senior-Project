# Basketball Shot Tracker App
# Developed by Christopher Hong
# Team Name: HoopIQ
# Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
# Start Web Development Date: October 2025
# Finished Web Development Date: June 2026 (Ideally)
# app/__init__.py

from flask import Flask
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
CORS(app)

# from app.routes.home import update
from app.routes.update import update_bp
app.register_blueprint(update_bp)

# from app.routes.stats import stats
from app.routes.stats import stats_bp
app.register_blueprint(stats_bp)

# from app.routes.frames import frames
from app.routes.frames import frames_bp
app.register_blueprint(frames_bp)

# from app.routes.health import health
from app.routes.health import health_bp
app.register_blueprint(health_bp)

# from app.routes.reset import reset
from app.routes.reset import reset_bp
app.register_blueprint(reset_bp)

# from app.routes.home import home
from app.routes.home import home_bp
app.register_blueprint(home_bp)