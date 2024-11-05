import streamlit as st

def about_page():
    # Page title and description
    st.markdown(
        """
        <div style='text-align: center; padding: 2em; background-color: #3c007a; color: white; border-radius: 10px;'>
            <h2 style='font-size: 2.5em;'>Our team</h2>
            <p style='font-size: 1.2em; line-height: 1.5;'>
                This project was carried out by highly skilled students from diverse backgrounds, including finance, engineering and more, all committed to applying their knowledge to a comprehensive and effective analysis of social network data.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

