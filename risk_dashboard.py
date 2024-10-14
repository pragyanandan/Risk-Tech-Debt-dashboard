import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os
import csv
import pandas as pd
from jira import JIRA

# Hardcoded valid credentials (for demo purposes; replace with secure handling in production)
VALID_USERS = {"admin": "password123", "anna": "ststwer!2", "callum": "ststwer!3","mark": "ststwer!4","nicola": "ststwer!5","rob": "ststwer!6", "abbas": "ststwer!7","denise": "ststwer!8"}

# Function to handle login
def login(username, password):
    if username in VALID_USERS and VALID_USERS[username] == password:
        st.session_state["authenticated"] = True
        st.session_state["user"] = username
        st.success(f"Welcome {username}!")
    else:
        st.error("Invalid username or password. Please try again.")

# Login screen
def login_screen():
    st.title("Login to Access the Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)

def fetch_jira_issues_and_create_csv(jira_server, jira_email, jira_api_token, filter_name="Prags-technology-Tech-Debt"):
    """
    Fetch issues from Jira using the specified filter, write the data to a CSV file, 
    and return the data as a Pandas DataFrame.
    
    Parameters:
    - jira_server: Jira server URL
    - jira_email: Jira user email for authentication
    - jira_api_token: Jira API token for authentication
    - filter_name: Name of the Jira filter to fetch issues (default is "Prags-technology-Tech-Debt")
    
    Returns:
    - DataFrame containing the fetched Jira issues
    """
    # Initialize Jira client
    jira = JIRA(server=jira_server, basic_auth=(jira_email, jira_api_token))

    try:
        # Get all favorite filters for the authenticated user
        all_filters = jira.favourite_filters()

        # Search for the filter by name
        target_filter = next((f for f in all_filters if f.name == filter_name), None)

        if target_filter:
            print(f"Filter '{filter_name}' found with ID: {target_filter.id}")
            
            # Fetch issues using the specified filter
            issues = jira.search_issues(f'filter={target_filter.id}', maxResults=100)

            # Define the column headers and field mapping for the CSV file
            columns = [
                "Key", "Summary", "Primary-Team", "FY-25", "FY-26", "FY-27", "FY-28", "FY-29",
                "Risk-Impact", "Risk-Likelihood", "risk-exposure-score", "IP-Platform-Scope", "MSI_Covered _Yes_No"
            ]

            # Create a list of dictionaries to hold the issue data
            data = []
            for issue in issues:
                # Create a row dictionary for each issue using custom fields
                row = {
                    "Key": issue.key,
                    "Summary": issue.fields.summary,
                    "Primary-Team": getattr(issue.fields, 'customfield_14208', "N/A"),
                    "FY-25": getattr(issue.fields, 'customfield_14215', "N/A"),
                    "FY-26": getattr(issue.fields, 'customfield_14216', "N/A"),
                    "FY-27": getattr(issue.fields, 'customfield_14217', "N/A"),
                    "FY-28": getattr(issue.fields, 'customfield_14218', "N/A"),
                    "FY-29": getattr(issue.fields, 'customfield_14353', "N/A"),
                    "Risk-Impact": getattr(issue.fields, 'customfield_14225', 0),
                    "Risk-Likelihood": getattr(issue.fields, 'customfield_14224', 0),
                    "risk-exposure-score": getattr(issue.fields, 'customfield_14226', 0),
                    "IP-Platform-Scope": getattr(issue.fields, 'customfield_14212', "N/A"),
                    "MSI_Covered _Yes_No": getattr(issue.fields, 'customfield_14354', "No")
                }
                # Append the row to the data list
                data.append(row)

            # Convert data to a DataFrame
            df = pd.DataFrame(data)

            # Specify the CSV file name
            csv_filename = "Tech-Dash.csv"

            # Check if the file already exists and notify the user
            if os.path.isfile(csv_filename):
                print(f"The file '{csv_filename}' already exists and will be overwritten.")
            else:
                print(f"Creating a new file: {csv_filename}")

            # Write the data to a CSV file with headers (overwrites if the file already exists)
            df.to_csv(csv_filename, index=False)

            print(f"Data successfully written to {csv_filename}")

            return df  # Return the data as a DataFrame
        else:
            print(f"Filter '{filter_name}' not found.")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def load_data(jira_server, jira_email,jira_api_token, fetch_jira):
    file_path = "Tech-Dash.csv"
    if fetch_jira or not os.path.exists(file_path):
        # Try fetching from Jira if fetch_jira is True or if CSV does not exist
        df = fetch_jira_issues_and_create_csv(jira_server, jira_email, jira_api_token)
        if df is not None:
            return df
        else:
            st.error("Jira fetch failed and no local file found.")
            return None
    else:
        # Load from local file if CSV exists
        return pd.read_csv(file_path)


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

# Function to preprocess data and calculate risk exposure
# Function to preprocess data and calculate risk exposure
def preprocess_data(df):
    #print("Processing data - in preprocess")

    # Define a helper function to safely extract numerical values from CustomFieldOption
    def get_numeric_value(field):
        if isinstance(field, (int, float)):
            return field  # Return if already a number
        elif hasattr(field, 'value'):  # Check if it's a CustomFieldOption with a 'value' attribute
            try:
                return float(field.value)  # Try to convert the 'value' to a number
            except ValueError:
                return 0  # Return 0 if conversion fails
        return 0  # Default to 0 if it's None or unrecognized
    
    # Apply the helper function to extract numerical values from 'Risk-Impact' and 'Risk-Likelihood'
    df['Risk-Impact'] = df['Risk-Impact'].apply(get_numeric_value)
    df['Risk-Likelihood'] = df['Risk-Likelihood'].apply(get_numeric_value)
    
    # Now that we have numerical values, fill missing values with 0
    df['Risk-Impact'].fillna(0, inplace=True)
    df['Risk-Likelihood'].fillna(0, inplace=True)
    
    # Calculate the risk-exposure-score
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
    def convert_to_string(value):
        if isinstance(value, list):
            return [str(v.value if hasattr(v, 'value') else v) for v in value]
        return str(value.value if hasattr(value, 'value') else value)

    filter_text = "<br>".join([f"<b>{key}:</b> {', '.join(convert_to_string(value))}" for key, value in filters.items() if value])

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
            'text': f'Tech Debt Risk Dashboard<br><sub>{filter_text}</sub>',
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
def generate_summary_table(df):
    # Ensure numeric columns are correctly converted to float
    df[['FY-25', 'FY-26', 'FY-27', 'FY-28', 'FY-29']] = df[['FY-25', 'FY-26', 'FY-27', 'FY-28', 'FY-29']].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Grouping by 'Primary-Team' and summing values for each FY column
    summary = df.groupby('Primary-Team')[['FY-25', 'FY-26', 'FY-27', 'FY-28', 'FY-29']].sum().reset_index()

    # Adding a "Total of the Row" column by summing each row's fiscal year values
    summary["Total of the Row"] = summary[['FY-25', 'FY-26', 'FY-27', 'FY-28', 'FY-29']].sum(axis=1)

    # Calculating Grand Totals for each FY and the Total of the Row
    grand_totals = summary[['FY-25', 'FY-26', 'FY-27', 'FY-28', 'FY-29', 'Total of the Row']].sum()
    grand_totals_row = pd.DataFrame(
        [['Grand Total'] + grand_totals.tolist()],
        columns=['Primary-Team', 'FY-25', 'FY-26', 'FY-27', 'FY-28', 'FY-29', 'Total of the Row']
    )

    # Concatenating the summary and grand total rows
    summary_table = pd.concat([summary, grand_totals_row], ignore_index=True)

    # Formatting the values to display them as currency
    summary_table.iloc[:, 1:] = summary_table.iloc[:, 1:].applymap(lambda x: f"${x:,.0f}" if x else "")

    return summary_table

def render_styled_table(summary_table):
    # Render the summary table with centered alignment using HTML/CSS
    st.markdown(
        """
        <style>
        table {
            width: 100%;
        }
        th, td {
            text-align: center;  /* Center-align all cells */
        }
        th:first-child, td:first-child {
            text-align: left;  /* Left-align the 'Primary Team' column */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.markdown with to_html() to render the DataFrame with styling
    st.markdown(summary_table.to_html(index=False, escape=False), unsafe_allow_html=True)


from IPython.display import display

import pandas as pd
import streamlit as st

import pandas as pd
import streamlit as st

def generate_detailed_table(df):
    # Select relevant columns
    columns = [
        "Key", "Summary", "FY-25", "FY-26", "FY-27", 
        "FY-28", "FY-29", "risk-exposure-score", 
        "IP-Platform-Scope", "MSI_Covered _Yes_No"
    ]

    # Filter the DataFrame to include only the specified columns
    detailed_table = df[columns].copy()

    # Fill NaN values with empty strings
    detailed_table = detailed_table.fillna("")

    # Format FY columns as currency with dollar sign and no decimals
    for col in ["FY-25", "FY-26", "FY-27", "FY-28", "FY-29"]:
        detailed_table[col] = detailed_table[col].apply(
            lambda x: f"${int(x):,}" if isinstance(x, (int, float)) else x
        )

    # Transform the 'Key' column values into hyperlinks
    detailed_table["Key"] = detailed_table["Key"].apply(
        lambda key: f'<a href="https://tvnztech.atlassian.net/browse/{key}" target="_blank">{key}</a>'
    )

    # Reset index to avoid displaying it
    detailed_table = detailed_table.reset_index(drop=True)

    # Convert DataFrame to HTML
    table_html = detailed_table.to_html(escape=False, index=False)

    # Custom CSS to align Summary column to the left
    st.markdown(
        """
        <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th {
            text-align: center;
            padding: 10px;
            cursor: pointer;
        }
        td {
            padding: 10px;
        }
        td:nth-child(2) {
            text-align: left !important;  /* Align Summary column to the left */
        }
        a {
            color: #1a73e8;
            text-decoration: none;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Display the HTML table
    st.markdown(table_html, unsafe_allow_html=True)

# Example usage:
# df = pd.read_csv("your_data.csv")
# generate_detailed_table(df)




def main_dashboard():
    # Jira server URL
   
    jira_server = os.getenv("JIRA_SERVER", "https://default-server-url.com")
    jira_email = os.getenv("JIRA_EMAIL", "default-email@example.com")
    jira_api_token = os.getenv("JIRA_API_TOKEN", "")
    '''
    jira_server = st.secrets["JIRA_SERVER"]
    jira_email = st.secrets["JIRA_EMAIL"]
    jira_api_token = st.secrets["JIRA_API_TOKEN"]
     '''

    st.sidebar.header("Actions")
    
    # Add a refresh button to trigger Jira fetch
    fetch_jira = st.sidebar.button("Refresh from Jira")
    
    # Call the load_data function, with fetch_jira controlling whether to fetch from Jira or use local data
    df = load_data(jira_server, jira_email,jira_api_token, fetch_jira=fetch_jira)

    if df is not None:
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

        # Generate and display the summary table
        st.subheader("Fiancial Summary Table")
        summary_table = generate_summary_table(filtered_df)
        #st.table(summary_table)  # Display the summary table
        render_styled_table(summary_table)  # Use the custom function to render

         # Generate and display the detailed table below the summary table
        st.subheader("Detailed Table")
        detailed_table = generate_detailed_table(filtered_df)
        st.dataframe(detailed_table, use_container_width=True)  # Display with a scrollable dataframe
    
    else:
        st.error("Failed to load data.")



# Main function to control login and dashboard
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        if st.sidebar.button("Logout"):
            st.session_state["authenticated"] = False
        else:
            main_dashboard()
    else:
        login_screen()


if __name__ == "__main__":
    main()