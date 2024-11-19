import streamlit as st
import pandas as pd
import settings
from EDA.eda import load_data, filter_data_by_date
from Visualization.visualization import (
    plot_platform_distribution,
    plot_interactions,
    plot_candidate_engagement_over_time,
    plot_candidate_comparison,
    plot_top_users,
    plot_avg_interactions_by_hour,
    plot_word_cloud
)

def dashboard_page():
    st.title("Social Media Data Dashboard")

    # Sidebar Configuration
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", settings.DEFAULT_START_DATE)
    end_date = st.sidebar.date_input("End Date", settings.DEFAULT_END_DATE)
    platform_filter = st.sidebar.multiselect("Select Platform(s)", settings.PLATFORMS, default=settings.PLATFORMS)

    # Load and filter data
    st.sidebar.text("Loading data...")
    data = load_data(settings.DATA_PATH, settings.FILES)
    filtered_data = filter_data_by_date(data, start_date, end_date)

    # Apply platform filter
    if platform_filter:
        filtered_data = filtered_data[filtered_data['platform'].isin(platform_filter)]

    # Dynamically populate candidates
    all_candidates = filtered_data['candidate_name'].dropna().unique().tolist()
    candidate_filter = st.sidebar.multiselect("Select Candidate(s)", options=all_candidates)

    # Apply candidate filter
    if candidate_filter:
        filtered_data = filtered_data[filtered_data['candidate_name'].isin(candidate_filter)]

    # Visualizations
    st.header("Visualizations")
    # 1. High-Level Overview
    plot_platform_distribution(filtered_data)

    # 2. Trends Over Time
    plot_interactions(filtered_data)
    plot_candidate_engagement_over_time(filtered_data)

    # 3. Candidate Insights
    plot_candidate_comparison(filtered_data)

    # 4. User Insights
    plot_top_users(filtered_data)

    # 5. Temporal Analysis
    plot_avg_interactions_by_hour(filtered_data)

    # 6. Qualitative Insights
    plot_word_cloud(filtered_data)
