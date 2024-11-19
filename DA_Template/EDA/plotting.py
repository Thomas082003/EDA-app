# plotting.py

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image
from io import BytesIO
import streamlit as st
import pytz
import matplotlib.dates as mdates
from itertools import cycle

def plot_combined_interactions_plotly(df, col_name, start_date, end_date, logo_image):
    if df.empty:
        print("No data available for the selected dates.")
        return

    # Convert 'datetime' column to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Ensure start_date and end_date are timezone-aware (using Mexico City timezone)
    start_date = start_date.tz_localize('America/Mexico_City')
    end_date = end_date.tz_localize('America/Mexico_City')

    # Filter data by date range
    df_filtered = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    #AQUI ERROR

    # Group by candidate and platform, then calculate total interactions
    daily_total_interactions = df_filtered.groupby(['candidate_name', 'platform'])[col_name].sum().reset_index()

    # Sort candidates by total interactions
    candidate_totals = daily_total_interactions.groupby('candidate_name')[col_name].sum().reset_index()
    candidate_totals = candidate_totals.sort_values(by=col_name, ascending=False)
    sorted_candidates = candidate_totals['candidate_name'].tolist()

    # Reorder dataframe based on sorted candidates
    daily_total_interactions['candidate_name'] = pd.Categorical(daily_total_interactions['candidate_name'], categories=sorted_candidates, ordered=True)
    daily_total_interactions = daily_total_interactions.sort_values('candidate_name')

    # Define a color map for platforms
    color_map = {
        'X': 'black',
        'Facebook': 'blue',
        'Youtube': 'red',
        'Instagram': 'magenta'
    }

    # Convert the PIL image to base64 to embed it in Plotly
    buffered = BytesIO()
    logo_image.save(buffered, format="PNG")
    logo_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Create total interaction bar chart
    fig = px.bar(daily_total_interactions,
                 x='candidate_name', y=col_name, color='platform',
                 color_discrete_map=color_map, barmode='group',
                 labels={col_name: 'Total Interactions', 'candidate_name': 'Candidate'},
                 title=f"Total Interactions for each Candidate by Platform ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

    # Update layout for the plot
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
        margin=dict(t=100),  # Increase top margin slightly
        height=600,  # Adjust height for better visibility
        width=1000
    )

    # Position the logo near the title
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{logo_base64}',
            xref="paper", yref="paper",
            x=0.01, y=1.18,  # Adjusted position for near-title placement
            sizex=0.17, sizey=0.17,
            xanchor="left", yanchor="top",
            layer="above"
        )
    )

    # Show the figure
    st.plotly_chart(fig)

def plot_top_8_peak_days(df, logo_path):
    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Convert to Mexico City timezone if needed
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Mexico_City')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('America/Mexico_City')

    # Drop rows where 'datetime' is NaT
    df = df.dropna(subset=['datetime'])

    # Filter only platform 'X' and data after April 1, 2024
    df = df[(df['platform'] == 'X') & (df['datetime'] >= '2024-04-01')]

    # Calculate daily interactions
    daily_interactions = df.groupby(df['datetime'].dt.date)['num_interaction'].sum()

    # Identify the 8 days with the highest interactions
    top_8_peak_days = daily_interactions.nlargest(8)

    # Define debate and election dates
    key_dates = {
        '2024-04-15': 'Debate 1',
        '2024-05-12': 'Debate 2',
        '2024-05-26': 'Debate 3',
        '2024-06-02': 'Election Day'
    }
    key_dates = {pd.Timestamp(date): label for date, label in key_dates.items()}

    # Set up Seaborn style
    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 8))

    # Plot daily interactions
    plt.plot(daily_interactions.index, daily_interactions.values, color='blue', linewidth=2)
    plt.scatter(top_8_peak_days.index, top_8_peak_days.values, color='#D0021B', s=100, label='Top 8 Peak Days', zorder=5, edgecolor='black')

    # Mark key dates
    for date, label in key_dates.items():
        if date in daily_interactions.index:
            plt.scatter(date, daily_interactions[date], color='purple', s=120, label=label, edgecolor='black')
            plt.annotate(f'{label}\n{daily_interactions[date]}', xy=(date, daily_interactions[date]), xytext=(0, 10),
                         textcoords='offset points', ha='center', color='purple', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lavender', alpha=0.8))

    # Annotate top 8 peak days
    for date, value in top_8_peak_days.items():
        plt.annotate(f'{date}\n{value}', xy=(date, value), xytext=(0, 10),
                     textcoords='offset points', ha='center', color='black', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8))

    # Titles and labels
    plt.title("Top 8 Peak Days of Interaction on X", fontsize=18, color='#333333')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Number of Interactions", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(visible=True, linestyle='--', alpha=0.7)

    # Add logo in the upper-left corner
    try:
        logo = Image.open(logo_path)
        newax = plt.gcf().add_axes([0.05, .9, 0.07, 0.10], anchor='NW', zorder=-1)
        newax.imshow(logo)
        newax.axis('off')
    except FileNotFoundError:
        print(f"Logo not found at {logo_path}. Skipping logo display.")
    except Exception as e:
        print(f"Error loading logo: {e}")

    # Display plot in Streamlit
    st.pyplot(plt.gcf())

def plot_curves_interactions(df, logo_path):
    # Load the logo as a PIL image
    try:
        logo = Image.open(logo_path)
    except FileNotFoundError:
        st.error("Logo file not found. Please check the path.")
        return

    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])  # Drop invalid rows

    # Extract only the date
    df['date'] = df['datetime'].dt.date

    # Filter only platform 'X'
    df = df[df['platform'] == 'X']

    # Define dates of interest and their titles
    debate_dates = ['2024-04-07', '2024-04-28', '2024-05-19', '2024-06-02']
    titles = ['1st Debate', '2nd Debate', '3rd Debate', 'Election Day']
    colors = ['green', 'blue', 'orange', 'red']  # Colors for debates and election

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Calculate the maximum number of interactions for scaling the y-axis
    max_interactions = 0
    for date in debate_dates:
        start_date = pd.to_datetime(date) - pd.Timedelta(days=3)
        end_date = pd.to_datetime(date) + pd.Timedelta(days=3)
        df_event = df[(df['date'] >= start_date.date()) & (df['date'] <= end_date.date())]
        daily_interactions = df_event.groupby('date')['num_interaction'].sum().reset_index()
        max_interactions = max(max_interactions, daily_interactions['num_interaction'].max())

    # Set y-axis limit with a margin
    y_limit = max_interactions * 1.1

    # Loop through each debate date
    for i, date in enumerate(debate_dates):
        start_date = pd.to_datetime(date) - pd.Timedelta(days=3)
        end_date = pd.to_datetime(date) + pd.Timedelta(days=3)
        df_event = df[(df['date'] >= start_date.date()) & (df['date'] <= end_date.date())]
        daily_interactions = df_event.groupby('date')['num_interaction'].sum().reset_index()

        # Fill missing dates in the range
        all_dates = pd.date_range(start=start_date, end=end_date).date
        daily_interactions = daily_interactions.set_index('date').reindex(all_dates, fill_value=0).reset_index()
        daily_interactions.columns = ['date', 'num_interaction']
        daily_interactions['weekday'] = pd.to_datetime(daily_interactions['date']).dt.day_name()

        # Plot data
        axes[i].fill_between(daily_interactions['weekday'], daily_interactions['num_interaction'], color=colors[i], alpha=0.3)
        axes[i].plot(daily_interactions['weekday'], daily_interactions['num_interaction'], marker='o', linestyle='-', color=colors[i], linewidth=2)

        # Highlight debate day
        event_date = pd.to_datetime(date).date()
        if event_date in daily_interactions['date'].values:
            event_value = daily_interactions[daily_interactions['date'] == event_date]['num_interaction'].values[0]
            axes[i].scatter(['Sunday'], [event_value], color='red', s=100, label=f"{titles[i]}")
            axes[i].annotate(f"{event_value}", xy=(daily_interactions[daily_interactions['date'] == event_date].index[0], event_value),
                             xytext=(0, 10), textcoords='offset points', ha='center', color='red', fontsize=10)

    # Main title and subtitle
    fig.suptitle('Interaction Patterns Throughout the Weeks of Mexico’s Four Key Election Events on X',
                 fontsize=20, fontweight='bold', color='black', y=0.98)
    plt.figtext(0.5, 0.93, 'Key Events: 1st Debate, 2nd Debate, 3rd Debate, and Election Day',
                ha='center', fontsize=14, color='gray')

    # Adjust layout for better display
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Add the logo in the top-left corner
    newax = fig.add_axes([0.02, 0.9, 0.1, 0.1], anchor='NW', zorder=-1)
    newax.imshow(logo)
    newax.axis('off')

    # Show the plot using Streamlit
    st.pyplot(fig)

def plot_candidate_hourly_heatmap(df, logo):
    # Extract hour and filter platform
    df['hour'] = df['datetime'].dt.hour
    df_x = df[df['platform'] == 'X']

    # Group by candidate and hour, calculate interactions
    candidate_hourly = df_x.groupby(['candidate_name', 'hour'])['num_interaction'].sum().unstack()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(candidate_hourly, cmap='coolwarm', annot=True, fmt=".1f", linewidths=.5)
    plt.title('Total Hourly Interactions on X by Candidate')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Candidate')

    # Show logo if available
    newax = plt.gcf().add_axes([0.05, 0.9, 0.1, 0.1], anchor='NW', zorder=-1)
    newax.imshow(logo)
    newax.axis('off')

    st.pyplot(plt.gcf())

def plot_hourly_trends_selected_weeks(df, logo_path):
    # Load logo
    logo = Image.open(logo_path)

    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter only platform 'X'
    df = df[df['platform'] == 'X']

    # Define the start date of the analysis period
    start_date = pd.to_datetime('2024-04-04')  # Thursday, April 4, 2024

    # Define the weeks to plot
    selected_weeks = [2, 3, 5, 6, 8]

    # Prepare the figure with subplots for each selected week
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for plot_index, week_number in enumerate(selected_weeks):
        # Calculate the week's range: Thursday to Wednesday
        week_start = start_date + pd.Timedelta(weeks=week_number-1)
        week_end = week_start + pd.Timedelta(days=6)

        # Convert dates to 'America/Mexico_City' timezone
        week_start = week_start.tz_localize('UTC').tz_convert('America/Mexico_City')
        week_end = week_end.tz_localize('UTC').tz_convert('America/Mexico_City')

        # Filter data for the current week
        df_week = df[(df['datetime'] >= week_start) & (df['datetime'] < week_end)]

        # Group interactions by hour of day
        hourly_interactions = df_week.groupby(df_week['datetime'].dt.hour)['num_interaction'].mean().reset_index()
        hourly_interactions.rename(columns={'datetime': 'hour'}, inplace=True)

        # Calculate average interactions for the week
        average_interactions = hourly_interactions['num_interaction'].mean()

        # Plot the interactions
        axes[plot_index].plot(hourly_interactions['hour'], hourly_interactions['num_interaction'],
                              marker='o', linestyle='-', color='blue', linewidth=2)

        # Annotate peak interactions
        if not hourly_interactions.empty:
            peak_interaction = hourly_interactions.loc[hourly_interactions['num_interaction'].idxmax()]
            peak_hour = peak_interaction['hour']
            peak_value = peak_interaction['num_interaction']
            percent_change = ((peak_value - average_interactions) / average_interactions) * 100 if average_interactions else 0
            axes[plot_index].annotate(f'{percent_change:.1f}%', xy=(peak_hour, peak_value), xytext=(peak_hour, peak_value + 300),
                                      textcoords='data', ha='center', color='red', fontsize=10)

        # Set titles and labels
        axes[plot_index].set_title(f'Week {week_number}: {week_start.strftime("%b %d")} - {week_end.strftime("%b %d")}', fontsize=14, fontweight='bold')
        axes[plot_index].set_xlabel('Hour of the Day', fontsize=12)
        axes[plot_index].set_ylabel('Average Interactions', fontsize=12)
        axes[plot_index].tick_params(axis='both', which='major', labelsize=10)

        # Set x-axis and y-axis limits
        axes[plot_index].set_xlim(0, 23)
        axes[plot_index].set_xticks(range(0, 24, 5))
        axes[plot_index].set_ylim(200, 3000)
        axes[plot_index].set_yticks(range(200, 3200, 200))
        axes[plot_index].grid(visible=True, linestyle='--', alpha=0.7)

    # Remove extra subplot if there are fewer than 6 weeks
    if len(selected_weeks) < len(axes):
        fig.delaxes(axes[-1])

    # General title and subtitle
    fig.suptitle('Hourly Interaction Trends in Selected Weeks', fontsize=24, fontweight='bold', color='black', y=0.98)
    plt.figtext(0.5, 0.93, 'Analysis of X Platform Interactions by Hour on Key Weeks', ha='center', fontsize=14, color='gray')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Display the plot using Streamlit
    st.pyplot(fig)
