import streamlit as st
from EDA.plotting import plot_top_8_peak_days, plot_curves_interactions, plot_hourly_trends_selected_weeks

def team_analysis_page(df, logo):
    st.write("### Top 8 Peak Days")
    plot_top_8_peak_days(df, logo)

    st.markdown(
        """
        <p style='font-size: 1.1em; color: white;'>
            The red dots highlight the eight peak interaction days, with some aligning with major events like debates and election day, 
            while others may be associated with additional events that also drove high interaction volumes.
        </p>
        """, unsafe_allow_html=True
    )

    st.write("### Interaction Patterns Throughout the Weeks of Mexico's Four Key Election Events on X")
    plot_curves_interactions(df, logo)

    st.markdown(
        """
        <p style='font-size: 1.1em; color: white;'>
            The four graphs indicate that the weeks of the debates saw peak engagement on Sundays, 
            supporting the observation that these major election events drive high levels of interaction early in the week as the public reacts to the debates.
        </p>
        """, unsafe_allow_html=True
    )

    st.write("### Hourly Interaction Trends in Selected Weeks")
    plot_hourly_trends_selected_weeks(df, logo)

    st.markdown(
        """
        <p style='font-size: 1.1em; color: white;'>
            As shown above, the standard weeks exhibit a similar pattern, indicating no significant increase around 8 PM, 
            which demonstrates normal behavior.
        </p>
        """, unsafe_allow_html=True
    )
