import streamlit as st
import pandas as pd
from .utils import load_data, selectDataByDate
from .plotting import plot_combined_interactions_plotly, plot_top_8_peak_days, plot_curves_interactions, plot_hourly_trends_selected_weeks
from PIL import Image

def eda_page():
    # Load logo image
    logo_path = "images/logo.jpeg"  # Adjust this path as necessary
    logo = Image.open(logo_path)

    st.header("Exploratory Data Analysis")

    # Define data path and load data
    path = '../data'  # Adjust the relative path based on your directory structure
    files = {
        'X': ['x/infotracer-x-abril-24.csv', 'x/infotracer-x-mayo-24.csv', 'x/infotracer-x-junio-24.csv', 'x/infotracer-x-julio-24.csv'],
        'Facebook': ['fb/infotracer-facebook-abril-24.csv', 'fb/infotracer-facebook-mayo-24.csv', 'fb/infotracer-facebook-junio-24.csv', 'fb/infotracer-facebook-julio-24.csv'],
        'Instagram': ['ig/infotracer-instagram-abril-24.csv', 'ig/infotracer-instagram-mayo-24.csv', 'ig/infotracer-instagram-junio-24.csv', 'ig/infotracer-instagram-julio-24.csv'],
        'Youtube': ['you/infotracer-youtube-abril-24.csv', 'you/infotracer-youtube-mayo-24.csv', 'you/infotracer-youtube-junio-24.csv', 'you/infotracer-youtube-julio-24.csv']
    }
    df = load_data(path, files)

    # Set the date range
    start_date = pd.to_datetime('2024-04-01 00:00:00')
    end_date = pd.to_datetime('2024-07-12 23:59:59')

    # Filter data
    df = selectDataByDate(df, start_date, end_date)

    # Display data summary
    st.write("### Top 8 Peak Days")

    # Call plot_top_8_peak_days
    plot_top_8_peak_days(df, logo)

    # Interpretation paragraph for Top 8 Peak Days plot
    st.markdown(
        """
        <p style='font-size: 1.1em; color: white;'>
            The red dots highlight the eight peak interaction days, with some aligning with major events like debates and election day, 
            while others may be associated with additional events that also drove high interaction volumes.
        </p>
        """, unsafe_allow_html=True
    )

    st.write("### Interaction Patterns Throughout the Weeks of Mexico's Four Key Election events on X")
    # Call plot_curves_interactions
    plot_curves_interactions(df, logo)

    # Interpretation paragraph for Curve Interactions plot
    st.markdown(
        """
        <p style='font-size: 1.1em; color: white;'>
            The four graphs indicate that the weeks of the debates saw peak engagement on Mondays, 
            supporting the observation that these major election events drive high levels of interaction early in the week as the public reacts to the debates.
        </p>
        """, unsafe_allow_html=True
    )

    st.write("### Hourly Interaction Trends in Selected Weeks")

    # Call plot_hourly_trends_selected_weeks
    plot_hourly_trends_selected_weeks(df, logo_path)

    # Interpretation paragraph for Hourly Trends Selected Weeks plot
    st.markdown(
        """
        <p style='font-size: 1.1em; color: white;'>
            As shown above, the standard weeks exhibit a similar pattern, indicating no significant increase around 8 PM, 
            which demonstrates normal behavior.
        </p>
        """, unsafe_allow_html=True
    )
