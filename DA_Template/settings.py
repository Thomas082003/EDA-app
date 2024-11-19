import os
import pandas as pd
import streamlit as st

# Default configuration for the app
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

FILES = {
    'X': ['x/infotracer-x-abril-24.csv', 'x/infotracer-x-mayo-24.csv', 'x/infotracer-x-junio-24.csv', 'x/infotracer-x-julio-24.csv'],
    'Facebook': ['fb/infotracer-facebook-abril-24.csv', 'fb/infotracer-facebook-mayo-24.csv', 'fb/infotracer-facebook-junio-24.csv', 'fb/infotracer-facebook-julio-24.csv'],
    'Instagram': ['ig/infotracer-instagram-abril-24.csv', 'ig/infotracer-instagram-mayo-24.csv', 'ig/infotracer-instagram-junio-24.csv', 'ig/infotracer-instagram-julio-24.csv'],
    'YouTube': ['you/infotracer-youtube-abril-24.csv', 'you/infotracer-youtube-mayo-24.csv', 'you/infotracer-youtube-junio-24.csv', 'you/infotracer-youtube-julio-24.csv']
}

DEFAULT_START_DATE = pd.to_datetime("2024-04-01")
DEFAULT_END_DATE = pd.to_datetime("2024-07-31")
PLATFORMS = ['Facebook', 'Instagram', 'X', 'YouTube']

# Apply dark mode CSS by default
def apply_dark_theme():
    css = """
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stButton>button {
            background-color: #1F1F1F;
            color: white;
            border: 1px solid #FAFAFA;
        }
        .stTextInput>div>input {
            background-color: #262626;
            color: white;
        }
        .block-container {
            background-color: #0E1117;
        }
    </style>
    """
    return css

# Apply the dark theme CSS globally
st.markdown(apply_dark_theme(), unsafe_allow_html=True)
