import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

color_map = {
            'X': '#A9A9A9',        # Light gray for X to match the dark theme
            'Facebook': '#3b5998',  # Facebook's blue
            'Instagram': '#C13584',  # Instagram's magenta
            'YouTube': '#FF0000',  # YouTube's red
}

candidate_color_map = {
    "Claudia Sheinbaum": "#800000",  # Deep maroon for Morena
    "Xóchitl Gálvez": "#0000FF",     # Bright blue for PAN
    "Jorge Álvarez Máynez": "#FF9900",  # Vibrant orange for MC
}

def plot_interactions(df):
    global color_map
    if 'datetime' in df and not df.empty:
        color_map = {
            'X': '#A9A9A9',        # Light gray for X to match the dark theme
            'Facebook': '#3b5998',  # Facebook's blue
            'Instagram': '#C13584',  # Instagram's magenta
            'YouTube': '#FF0000',  # YouTube's red
        }
        fig = px.line(
            df,
            x='datetime',
            y='num_interaction',
            color='platform',
            title="Total Interactions Over Time",
            labels={'datetime': 'Date', 'num_interaction': 'Total Interactions'},
            color_discrete_map=color_map  # Apply the color map
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for this visualization.")


def plot_candidate_comparison(df):
    if 'candidate_name' in df and not df.empty:
        fig = px.bar(
            df,
            x='candidate_name',
            y='num_interaction',
            color='candidate_name',
            title="Total Interactions by Candidate",
            labels={'candidate_name': 'Candidate', 'num_interaction': 'Total Interactions'},
            color_discrete_map=candidate_color_map  # Apply candidate colors
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for this visualization.")


def plot_platform_distribution(df):
    global color_map
    if 'platform' in df and not df.empty:
        color_map = {
            'X': '#A9A9A9',        # Light gray for X to match the dark theme
            'Facebook': '#3b5998',  # Facebook's blue
            'Instagram': '#C13584',  # Instagram's magenta
            'YouTube': '#FF0000',  # YouTube's red
        }
        fig = px.pie(
            df,
            names='platform',
            values='num_interaction',
            title="Interaction Distribution by Platform",
            labels={'platform': 'Platform', 'num_interaction': 'Total Interactions'},
            color_discrete_map=color_map  # Apply the color map
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for this visualization.")

def plot_top_users(df):
    if 'username' in df and not df.empty:
        top_users = (
            df.groupby('username')['num_interaction']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            top_users,
            x='num_interaction',
            y='username',
            orientation='h',
            title="Top 10 Users by Interactions",
            labels={'num_interaction': 'Total Interactions', 'username': 'User'},
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for this visualization.")


def plot_candidate_engagement_over_time(df):
    if 'datetime' in df and 'candidate_name' in df and not df.empty:
        fig = px.line(
            df,
            x='datetime',
            y='num_interaction',
            color='candidate_name',
            title="Interactions by Candidate Over Time",
            labels={'datetime': 'Date', 'num_interaction': 'Total Interactions'},
            color_discrete_map=candidate_color_map
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for this visualization.")

def plot_word_cloud(df):
    if 'text' in df and not df.empty:
        text_data = ' '.join(df['text'].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_data)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Most Common Words in Posts")
        st.pyplot(plt)
    else:
        st.write("No text data available for this visualization.")

def plot_avg_interactions_by_hour(df):
    if 'datetime' in df and not df.empty:
        # Ensure a copy of the DataFrame is used
        df = df.copy()

        # Extract hour from datetime
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        
        # Calculate average interactions by hour
        avg_interactions = (
            df.groupby('hour')['num_interaction']
            .mean()
            .reset_index()
        )
        
        # Create line plot
        fig = px.line(
            avg_interactions,
            x='hour',
            y='num_interaction',
            title="Average Interactions by Hour",
            labels={'hour': 'Hour of the Day', 'num_interaction': 'Average Interactions'},
            markers=True
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for this visualization.")

