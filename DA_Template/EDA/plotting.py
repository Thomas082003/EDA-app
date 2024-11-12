import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Ensure datetime conversions for timezone handling
import pytz

# Helper function to convert PIL image to base64
def pil_image_to_base64(logo_image):
    buffered = BytesIO()
    logo_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def plot_combined_interactions_plotly(df, col_name, start_date, end_date, logo_image):
    if df.empty:
        print("No data available for the selected dates.")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    start_date = start_date.tz_localize('America/Mexico_City')
    end_date = end_date.tz_localize('America/Mexico_City')
    df_filtered = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    daily_total_interactions = df_filtered.groupby(['candidate_name', 'platform'])[col_name].sum().reset_index()
    candidate_totals = daily_total_interactions.groupby('candidate_name')[col_name].sum().reset_index()
    candidate_totals = candidate_totals.sort_values(by=col_name, ascending=False)
    sorted_candidates = candidate_totals['candidate_name'].tolist()

    daily_total_interactions['candidate_name'] = pd.Categorical(daily_total_interactions['candidate_name'], categories=sorted_candidates, ordered=True)
    daily_total_interactions = daily_total_interactions.sort_values('candidate_name')

    color_map = {
        'X': 'black',
        'Facebook': 'blue',
        'Youtube': 'red',
        'Instagram': 'magenta'
    }

    logo_base64 = pil_image_to_base64(logo_image)

    fig = px.bar(daily_total_interactions,
                 x='candidate_name', y=col_name, color='platform',
                 color_discrete_map=color_map, barmode='group',
                 labels={col_name: 'Total Interactions', 'candidate_name': 'Candidate'},
                 title=f"Total Interactions for each Candidate by Platform ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

    fig.update_layout(
        title={
            'text': 'Total Interactions for each Candidate by Platform<br><sup>Exploring interaction patterns across different platforms</sup>',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'y': 0.90,
            'font': {'size': 18, 'color': 'black'}
        },
        xaxis_title='Candidate Name',
        yaxis_title='Total Interactions',
        legend_title='Platform',
        margin=dict(t=100),
        height=600,
        width=1000
    )

    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{logo_base64}',
            xref="paper", yref="paper",
            x=0.01, y=1.18,
            sizex=0.17, sizey=0.17,
            xanchor="left", yanchor="top",
            layer="above"
        )
    )

    st.plotly_chart(fig)

def plot_top_8_peak_days(df, logo):
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('America/Mexico_City')
    df = df[(df['datetime'] >= '2024-04-01') & (df['platform'] == 'X')]
    daily_interactions = df.groupby(df['datetime'].dt.date)['num_interaction'].sum()
    top_8_peak_days = daily_interactions.nlargest(8)

    key_dates = {
        '2024-04-15': 'Debate 1',
        '2024-05-12': 'Debate 2',
        '2024-05-26': 'Debate 3',
        '2024-06-02': 'Election Day'
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_interactions.index, y=daily_interactions.values,
        mode='lines+markers', name='Daily Interactions'
    ))

    fig.add_trace(go.Scatter(
        x=top_8_peak_days.index, y=top_8_peak_days.values,
        mode='markers', name='Top 8 Peak Days',
        marker=dict(color='red', size=10, line=dict(color='black', width=2))
    ))

    for date, label in key_dates.items():
        if pd.Timestamp(date) in daily_interactions.index:
            fig.add_annotation(
                x=date, y=daily_interactions[date],
                text=label, showarrow=True, arrowhead=2
            )

    fig.update_layout(
        title="Interaction Trends with Top Peak Days",
        xaxis_title="Date",
        yaxis_title="Number of Interactions",
        height=600,
        width=1000
    )

    st.plotly_chart(fig)

def plot_candidate_hourly_heatmap(df, logo):
    df['hour'] = df['datetime'].dt.hour
    df_x = df[df['platform'] == 'X']
    candidate_hourly = df_x.groupby(['candidate_name', 'hour'])['num_interaction'].sum().unstack().fillna(0)

    fig = px.imshow(candidate_hourly, aspect='auto', color_continuous_scale='coolwarm',
                    labels={'color': 'Total Interactions'}, title='Hourly Interactions by Candidate')
    fig.update_xaxes(title="Hour of the Day")
    fig.update_yaxes(title="Candidate")

    st.plotly_chart(fig)

def plot_hourly_trends_selected_weeks(df, logo):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['platform'] == 'X']

    start_date = pd.to_datetime('2024-04-04')
    selected_weeks = [2, 3, 5, 6, 8]

    fig = go.Figure()

    for week_number in selected_weeks:
        week_start = start_date + pd.Timedelta(weeks=week_number-1)
        week_end = week_start + pd.Timedelta(days=6)
        week_start = week_start.tz_localize('UTC').tz_convert('America/Mexico_City')
        week_end = week_end.tz_localize('UTC').tz_convert('America/Mexico_City')

        df_week = df[(df['datetime'] >= week_start) & (df['datetime'] < week_end)]
        hourly_interactions = df_week.groupby(df_week['datetime'].dt.hour)['num_interaction'].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=hourly_interactions['datetime'], y=hourly_interactions['num_interaction'],
            mode='lines+markers', name=f'Week {week_number}'
        ))

    fig.update_layout(
        title="Hourly Interaction Trends in Selected Weeks",
        xaxis_title="Hour of the Day",
        yaxis_title="Average Interactions",
        height=600,
        width=1000
    )

    st.plotly_chart(fig)
