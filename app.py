"""
VRP Prediction Dashboard - Root Entry Point
Wrapper for dashboard/app.py
"""
import subprocess
import sys
import os

# Change to the project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to path
sys.path.insert(0, '.')

# Import and run the dashboard app
from dashboard.app import *
