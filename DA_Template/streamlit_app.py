import streamlit as st
from EDA.eda import eda_page
from Background.intro import intro_page
from Background.facts import facts_page
from Background.about import about_page
from Background.example import copyright_page

st.set_page_config(page_title="Data Analytics App", page_icon=":bar_chart:")

# Title and logo
st.title("Data Analytics")

# Define pages in the dictionary
page_dict = {
    "Introduction": intro_page,
    "EDA": eda_page,
    "About Us": about_page,
}

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Choose a page", list(page_dict.keys()))

# Run the selected page function
if selected_page in page_dict:
    page_dict[selected_page]()
