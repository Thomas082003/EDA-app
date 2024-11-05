import streamlit as st
from PIL import Image

def intro_page():
    # Title and main text
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; background-color: #3c007a; color: white; border-radius: 10px;'>
            <h2 style='font-size: 2em;'>Research Objective</h2>
            <p style='font-size: 1.1em; line-height: 1.5;'>
                This research aims to collect and analyze data from major social media platforms, such as Instagram, Facebook, YouTube and X, covering the period from January 1, 2024 to July 31, 2024, with a focus on Mexico's 2024 national elections.
            </p>
            <p style='font-size: 1.1em; line-height: 1.5;'>
                The objective of this research is to identify and analyze the misinformation present on social media platforms. From these findings, we seek to develop effective strategies for our training partner.
            </p>
        </div>
        """, unsafe_allow_html=True
    )