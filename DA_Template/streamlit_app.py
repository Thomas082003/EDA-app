import streamlit as st
import os
from Background import intro, about, team_analysis, dashboard, sentiment_analysis
from EDA.eda import load_data
import settings

# Sidebar navigation
st.sidebar.title("Shaping the Nation: 2024 Electoral Sentiment Insights")
page = st.sidebar.radio(
    "Navigate to:",
    ["Introduction", "About Us", "Team Analysis", "Dashboard", "Sentiment Analysis"]
)

# Attempt to load data
try:
    df = load_data(settings.DATA_PATH, settings.FILES)
except Exception as e:
    st.error(f"Error loading data: {e}")

# Debugging: Ensure the logo path exists
logo = "images/logo.jpeg"  # Replace with your logo path or object
if not os.path.exists(logo):
    st.error(f"Logo not found: {logo}")
    
# Page selection logic
if page == "Introduction":
    intro.intro_page()
elif page == "About Us":
    about.about_page()
elif page == "Team Analysis":
    if df is not None and not df.empty and logo:
        team_analysis.team_analysis_page(df, logo)
    else:
        st.error("Data or logo not loaded.")
elif page == "Dashboard":
    dashboard.dashboard_page()
elif page == "Sentiment Analysis":  # Add this block
    sentiment_analysis.sentiment_analysis_page()