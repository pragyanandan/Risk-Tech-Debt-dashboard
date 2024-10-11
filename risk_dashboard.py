import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Page configuration to start in wide/full-screen layout
st.set_page_config(layout="wide")

# Custom CSS to make the chart appear in a full-screen-like effect
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        max-width: 95%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the data
file_path = "Test-Dash.csv"
df = pd.read_csv(file_path)

# Function to preprocess data and calculate risk exposure
def preprocess_data(df):
    df['Risk-Impact'].fillna(0, inplace=True)
    df['Risk-Likelihood'].fillna(0, inplace=True)
    df['risk-exposure-score'] = df['Risk-Impact'] * df['Risk-Likelihood']
    return df

# Function to generate risk matrix based on likelihood and impact
def generate_risk_matrix(df):
    risk_matrix = pd.DataFrame(index=["Rare", "Unlikely", "Possible", "Likely", "Almost Certain"],
                               columns=["Minor", "Moderate", "Significant", "Major", "Severe"])
    likelihood_mapping = {1: "Rare", 2: "Unlikely", 3: "Possible", 4: "Likely", 5: "Almost Certain"}
    impact_mapping = {1: "Minor", 2: "Moderate", 3: "Significant", 4: "Major", 5: "Severe"}
    
    for _, row in df.iterrows():
        likelihood = row['Risk-Likelihood']
        impact = row['Risk-Impact']
        summary = row['Summary']
        likelihood_str = likelihood_mapping.get(likelihood, "")
        impact_str = impact_mapping.get(impact, "")
        
        if pd.notna(risk_matrix.loc[likelihood_str, impact_str]):
            risk_matrix.loc[likelihood_str, impact_str] += "\n" + summary
        else:
            risk_matrix.loc[likelihood_str, impact_str] = summary
            
    return risk_matrix

# Function to create a Plotly heatmap with a custom color scale and conditional risk score annotations
def plot_risk_matrix(risk_matrix, filters, show_scores):
    custom_colorscale = [
        [0.0, 'rgb(0, 128, 0)'],      # Dark Green for low risk
        [0.2, 'rgb(173, 255, 47)'],    # Yellow-Green for moderate risk
        [0.4, 'rgb(255, 255, 0)'],     # Yellow for significant risk
        [0.6, 'rgb(255, 165, 0)'],     # Orange for major risk
        [0.8, 'rgb(255, 69, 0)'],      # Orange-Red for severe risk
        [1.0, 'rgb(255, 0, 0)']        # Red for extreme risk
    ]

    # Calculate risk scores for each cell
    risk_scores = [[(y + 1) * (x + 1) for x in range(5)] for y in range(5)]

    # Create Heatmap
    trace = go.Heatmap(
        z=risk_scores,
        x=['Minor', 'Moderate', 'Significant', 'Major', 'Severe'],
        y=['Rare', 'Unlikely', 'Possible', 'Likely', 'Almost Certain'],
        colorscale=custom_colorscale,
        showscale=False,
        text=risk_matrix.values,
        texttemplate='%{text}',
        hoverinfo='text'
    )
    
    # Create a formatted string of applied filters to show in the title
    filter_text = "<br>".join([f"<b>{key}:</b> {', '.join(value)}" for key, value in filters.items() if value])

    # Layout and Main Annotations for Summaries
    annotations = [
        dict(
            x=x, y=y,
            text=str(risk_matrix.iloc[y, x]).replace("\n", "<br>") if pd.notna(risk_matrix.iloc[y, x]) else "",
            xref='x1', yref='y1',
            showarrow=False,
            font=dict(color='black', size=12),
            align='center'
        ) for y in range(len(risk_matrix.index)) for x in range(len(risk_matrix.columns))
    ]

    # Conditionally include risk score annotations if the checkbox is enabled
    if show_scores:
        score_annotations = [
            dict(
                x=x, y=y,
                text=str(risk_scores[y][x]),
                xref='x1', yref='y1',
                showarrow=False,
                font=dict(color='white', size=14, weight='bold'),
                align='center',
                xanchor='right',
                yanchor='top',
                ax=-10, ay=-10
            ) for y in range(len(risk_matrix.index)) for x in range(len(risk_matrix.columns))
        ]
        annotations += score_annotations  # Add risk score annotations if enabled

    layout = go.Layout(
        title={
            'text': f'TVNZ Tech Debt Risk Dashboard<br><sub>{filter_text}</sub>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(title='Impact', tickangle=-45, showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(title='Likelihood', showline=True, linewidth=2, linecolor='black', mirror=True),
        height=900,
        width=1400,
        font=dict(family='Arial', size=14, color='black'),
        margin=dict(l=30, r=30, t=150, b=100),
        annotations=annotations  # Include or exclude risk score annotations based on the toggle
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig

def main():
    st.title("TVNZ Tech Debt Risk Dashboard")

    # Preprocess the data
    df_processed = preprocess_data(df)

    # Sidebar filters and toggle button for showing risk scores
    st.sidebar.header("Filters")

    # New Primary-Team filter
    primary_team_filter = st.sidebar.multiselect("Select Primary Team", options=df_processed['Primary-Team'].unique(), default=df_processed['Primary-Team'].unique())
    
    # Existing filters
    ip_platform_filter = st.sidebar.multiselect("IP Platform Scope?", options=df_processed['IP-Platform-Scope'].unique(), default=df_processed['IP-Platform-Scope'].unique())
    msi_covered_filter = st.sidebar.multiselect("Is it part of Accenture Business Case?", options=df_processed['MSI_Covered _Yes_No'].unique(), default=df_processed['MSI_Covered _Yes_No'].unique())
    
    # Add a checkbox toggle to control risk score display
    show_scores = st.sidebar.checkbox("Show Risk Scores", value=True)

    # Apply filters
    filtered_df = df_processed[
        (df_processed['Primary-Team'].isin(primary_team_filter)) &
        (df_processed['IP-Platform-Scope'].isin(ip_platform_filter)) &
        (df_processed['MSI_Covered _Yes_No'].isin(msi_covered_filter))
    ]

    # Create a dictionary of applied filters
    applied_filters = {
        "Primary Team": primary_team_filter,
        "IP Platform": ip_platform_filter,
        "MSI Covered": msi_covered_filter
    }

    # Generate and plot the risk matrix for filtered data
    risk_matrix = generate_risk_matrix(filtered_df)
    st.plotly_chart(plot_risk_matrix(risk_matrix, applied_filters, show_scores), use_container_width=True)

if __name__ == "__main__":
    main()
