import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

warnings.filterwarnings('ignore')

# Set page configuration with a logo
st.set_page_config(page_title="FIFA", page_icon=":bar_chart:", layout="wide")

# Include a large logo image at the top right
logo_url = "https://staticg.sportskeeda.com/editor/2020/06/a43d4-15928015256443-800.jpg"
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1 style="flex: 1;"> :bar_chart: FIFA DATASET PROJECT - Group 6</h1>
        <img src="{logo_url}" style="width: 100px; height: auto;" />
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.markdown("Navigate to: [Player Comparison Dashboard](http://localhost:8501)")

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding="ISO-8859-1")
    else:
        return pd.read_excel(file, engine='openpyxl')

# Upload file
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "xlsx", "xls"])

if fl is not None:
    df = load_data(fl)
else:
    df = pd.read_csv("data_22_23_4.csv", encoding="ISO-8859-1")

# Check if DataFrame is empty
if df.empty:
    st.error("Uploaded file is empty or couldn't be loaded.")
else:
    # Data Summary
    st.header("Data Summary")
    st.write(f"**Number of Players:** {df.shape[0]}")
    st.write(f"**Number of Features:** {df.shape[1]}")
    st.write(f"**Unique Club Positions:** {df['club_position'].nunique()}")
    st.write(f"**Unique Leagues:** {df['league_name'].nunique()}")
    st.write(f"**Average Overall Rating:** {df['overall'].mean():.2f}")

    # Checkboxes to show top players
    col1, col2 = st.columns(2)
    
    with col1:
        show_top_positions = st.checkbox("Show Top Player per Club Position")
    
    with col2:
        show_top_leagues = st.checkbox("Show Top Player per League")
    
    if show_top_positions or show_top_leagues:
        st.header("Top Players Visualization")

    if show_top_positions:
        st.subheader("Top Player per Club Position")
        unique_positions = df['club_position'].unique()
        for position in unique_positions:
            position_df = df[df['club_position'] == position]
            top_player = position_df.nlargest(1, 'overall').iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                try:
                    response = requests.get(top_player['player_face_url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=f"{top_player['short_name']} (Overall: {top_player['overall']})", width=100)
                except UnidentifiedImageError:
                    st.write(f"Image could not be loaded for player: {top_player['short_name']}")
                except requests.exceptions.RequestException as e:
                    st.write(f"Error fetching image for player: {top_player['short_name']}, Error: {str(e)}")
            
            with col2:
                st.write(f"**Player Name:** {top_player['short_name']}")
                st.write(f"**Overall Rating:** {top_player['overall']}")
                st.write(f"**Club Name:** {top_player['club_name']}")
                st.write(f"**Age:** {top_player['age']}")
                st.write(f"**Height:** {top_player['height_cm']} cm")
                st.write(f"**Weight:** {top_player['weight_kg']} kg")
                st.write(f"**Pace:** {top_player['pace']}")
                st.write(f"**Shooting:** {top_player['shooting']}")
                st.write(f"**Passing:** {top_player['passing']}")
                st.write(f"**Dribbling:** {top_player['dribbling']}")
                st.write(f"**Defending:** {top_player['defending']}")
                st.write(f"**Physic:** {top_player['physic']}")
                
    if show_top_leagues:
        st.subheader("Top Player per League")
        unique_leagues = df['league_name'].unique()
        for league in unique_leagues:
            league_df = df[df['league_name'] == league]
            top_player = league_df.nlargest(1, 'overall').iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                try:
                    response = requests.get(top_player['player_face_url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=f"{top_player['short_name']} (Overall: {top_player['overall']})", width=100)
                except UnidentifiedImageError:
                    st.write(f"Image could not be loaded for player: {top_player['short_name']}")
                except requests.exceptions.RequestException as e:
                    st.write(f"Error fetching image for player: {top_player['short_name']}, Error: {str(e)}")
            
            with col2:
                st.write(f"**Player Name:** {top_player['short_name']}")
                st.write(f"**Overall Rating:** {top_player['overall']}")
                st.write(f"**Club Name:** {top_player['club_name']}")
                st.write(f"**Age:** {top_player['age']}")
                st.write(f"**Height:** {top_player['height_cm']} cm")
                st.write(f"**Weight:** {top_player['weight_kg']} kg")
                st.write(f"**Pace:** {top_player['pace']}")
                st.write(f"**Shooting:** {top_player['shooting']}")
                st.write(f"**Passing:** {top_player['passing']}")
                st.write(f"**Dribbling:** {top_player['dribbling']}")
                st.write(f"**Defending:** {top_player['defending']}")
                st.write(f"**Physic:** {top_player['physic']}")

    # Optional: Add a regression analysis example
    st.sidebar.header("Regression Analysis")
    if st.sidebar.button("Run Regression Analysis"):
        st.sidebar.write("Performing regression analysis...")

        # Feature selection for regression
        features = ['age', 'height_cm', 'weight_kg', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        df_reg = df[features + ['overall']].dropna()

        X = df_reg[features]
        y = df_reg['overall']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("**Regression Model Performance**")
        st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**R^2 Score:** {r2_score(y_test, y_pred):.2f}")

        # Display coefficients
        coef_df = pd.DataFrame(model.coef_, index=features, columns=['Coefficient'])
        st.write("**Regression Coefficients**")
        st.write(coef_df)

    # Function to generate random colors
    def random_colors(num_colors):
        np.random.seed(0)  # For reproducibility
        return [f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})' for _ in range(num_colors)]

    # Visualizations
    st.header("Visual Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Overall Ratings")
        fig1 = px.histogram(df, x="overall", title="Distribution of Overall Player Ratings",
                            color_discrete_sequence=random_colors(1))
        st.plotly_chart(fig1)

        # Club Positions Distribution
        st.subheader("Club Positions Distribution")
        positions_df = df[~df['club_position'].isin(['SUB', 'RES'])]['club_position'].value_counts().reset_index()
        positions_df.columns = ['Position', 'Count']
        fig3 = px.pie(positions_df, names='Position', values='Count', title="Club Positions Distribution",
                    color_discrete_sequence=random_colors(len(positions_df)))
        st.plotly_chart(fig3)

    with col2:
        # League-wise Average Overall Ratings
        st.subheader("League-wise Average Overall Ratings")
        league_avg_df = df.groupby('league_name')['overall'].mean().reset_index()
        league_avg_df = league_avg_df.sort_values(by='overall', ascending=False)
        fig2 = px.bar(league_avg_df, x='league_name', y='overall', title="League-wise Average Overall Ratings",
                    color_discrete_sequence=random_colors(len(league_avg_df)))
        st.plotly_chart(fig2)

        # Height vs Overall Rating
        st.subheader("Height vs Overall Rating")
        fig4 = px.scatter(df, x="height_cm", y="overall", trendline="ols", 
                    title="Relationship between Height and Overall Rating",
                    color_discrete_sequence=random_colors(1))
    st.plotly_chart(fig4)

    st.markdown("### Key Insights")
    st.markdown("- **Distribution of Overall Ratings:** The majority of players have ratings between 60 and 80, indicating a balanced distribution of skill levels.")
    st.markdown("- **Club Positions Distribution:** Most players occupy the midfield and forward positions, with defenders and goalkeepers being less common.")
    st.markdown("- **League-wise Average Ratings:** Top leagues like La Liga, Premier League, and Bundesliga have the highest average overall ratings, reflecting the concentration of top talent.")
    st.markdown("- **Height vs Overall Rating:** There is a slight positive correlation between a player's height and their overall rating, indicating that taller players might have a slight advantage in terms of overall performance.")
