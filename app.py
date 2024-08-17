import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# Set page config
st.set_page_config(page_title="FIFA Player Analysis Dashboard", page_icon="⚽", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">FIFA Player Analysis Dashboard</p>', unsafe_allow_html=True)

st.markdown("Navigate to: [FIFA DASHBOARD](http://localhost:8502)")

# Load data
@st.cache_data
def load_data():
    file_path = r"C:\Users\aashi\Downloads\archive\data_22_23_4.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        return df
    else:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
fifa_version = st.sidebar.multiselect("FIFA Version", df["fifa_version"].unique())
leagues = st.sidebar.multiselect("League", df["league_name"].unique())
clubs = st.sidebar.multiselect("Club", df["club_name"].unique())
positions = st.sidebar.multiselect("Position", df["player_positions"].unique())
age_range = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (18, 40))

# Apply filters
if fifa_version:
    df = df[df["fifa_version"].isin(fifa_version)]
if leagues:
    df = df[df["league_name"].isin(leagues)]
if clubs:
    df = df[df["club_name"].isin(clubs)]
if positions:
    df = df[df["player_positions"].isin(positions)]
df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

# Main content
st.header("Player Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Players", len(df))
col2.metric("Average Overall Rating", f"{df['overall'].mean():.2f}")
col3.metric("Average Potential", f"{df['potential'].mean():.2f}")
col4.metric("Average Value (EUR)", f"{df['value_eur'].mean():,.0f}")

# Radar Chart of Top Player
st.subheader("Top Player Analysis")
top_player = df.loc[df["overall"].idxmax()]
radar_attributes = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
radar_values = [top_player[attr] for attr in radar_attributes]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=radar_values,
    theta=radar_attributes,
    fill='toself',
    name=top_player["short_name"]
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )),
    showlegend=False
)

st.plotly_chart(fig_radar)

st.write(f"Top Player: {top_player['short_name']} (Overall: {top_player['overall']})")

# Player Attributes Distribution
st.subheader("Player Attributes Distribution")
attributes = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
attribute_to_plot = st.selectbox("Select Attribute", attributes)

fig_dist = px.histogram(df, x=attribute_to_plot, color="player_positions", 
                        marginal="box", hover_data=["short_name", "overall"])
fig_dist.update_layout(
    title=f"Distribution of {attribute_to_plot.capitalize()}",
    xaxis_title=attribute_to_plot.capitalize(),
    yaxis_title="Count",
    legend_title="Position"
)
st.plotly_chart(fig_dist)

# Correlation Heatmap
st.subheader("Attribute Correlation Heatmap")
corr_attributes = attributes + ["overall", "potential", "value_eur", "wage_eur", "age"]
corr_df = df[corr_attributes].corr()

fig_heatmap = px.imshow(corr_df, text_auto=True, aspect="auto", 
                        color_continuous_scale="Viridis")
fig_heatmap.update_layout(title="Correlation Heatmap of Player Attributes")
st.plotly_chart(fig_heatmap)

# Player Comparison
st.header("Player Comparison")
players_to_compare = st.multiselect("Select players to compare", df["short_name"].unique(), max_selections=3)

if players_to_compare:
    comparison_df = df[df["short_name"].isin(players_to_compare)]
    fig_comparison = go.Figure()

    for player in players_to_compare:
        player_data = comparison_df[comparison_df["short_name"] == player]
        fig_comparison.add_trace(go.Scatterpolar(
            r=[player_data[attr].values[0] for attr in attributes],
            theta=attributes,
            fill='toself',
            name=player
        ))

    fig_comparison.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True
    )

    st.plotly_chart(fig_comparison)

# League Analysis
st.header("League Analysis")

league_stats = df.groupby("league_name").agg({
    "overall": "mean",
    "potential": "mean",
    "value_eur": "mean",
    "wage_eur": "mean"
}).reset_index()

fig_league = px.scatter(league_stats, x="overall", y="potential", size="value_eur", color="wage_eur",
                        hover_name="league_name", log_x=True, size_max=60)

fig_league.update_layout(
    title="League Comparison: Overall vs Potential",
    xaxis_title="Average Overall Rating",
    yaxis_title="Average Potential",
    coloraxis_colorbar_title="Average Wage (EUR)"
)

st.plotly_chart(fig_league)

# Age vs Overall Rating
st.subheader("Age vs Overall Rating")
fig_age_overall = px.scatter(df, x="age", y="overall", color="player_positions", 
                            hover_data=["short_name", "club_name", "value_eur"])
fig_age_overall.update_layout(
    title="Age vs Overall Rating",
    xaxis_title="Age",
    yaxis_title="Overall Rating"
)
st.plotly_chart(fig_age_overall)

# Value Prediction Model
st.header("Player Value Prediction Model")

features = ["overall", "potential", "age", "wage_eur"] + attributes
X = df[features]
y = df["value_eur"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write(f"Model R-squared: {r2_score(y_test, y_pred):.4f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.coef_
})
feature_importance = feature_importance.sort_values("Importance", ascending=False)

fig_importance = px.bar(feature_importance, x="Importance", y="Feature", orientation="h")
fig_importance.update_layout(title="Feature Importance for Player Value Prediction")
st.plotly_chart(fig_importance)

# Player Value Predictor
st.subheader("Predict Player Value")

col1, col2 = st.columns(2)

with col1:
    pred_overall = st.slider("Overall Rating", 40, 100, 75)
    pred_potential = st.slider("Potential", 40, 100, 80)
    pred_age = st.slider("Age", 16, 45, 25)
    pred_wage = st.number_input("Wage (EUR)", 0, 1000000, 50000)

with col2:
    pred_pace = st.slider("Pace", 0, 100, 70)
    pred_shooting = st.slider("Shooting", 0, 100, 70)
    pred_passing = st.slider("Passing", 0, 100, 70)
    pred_dribbling = st.slider("Dribbling", 0, 100, 70)
    pred_defending = st.slider("Defending", 0, 100, 70)
    pred_physic = st.slider("Physic", 0, 100, 70)

if st.button("Predict Value"):
    pred_features = [pred_overall, pred_potential, pred_age, pred_wage, 
                    pred_pace, pred_shooting, pred_passing, pred_dribbling, pred_defending, pred_physic]
    predicted_value = model.predict([pred_features])[0]
    st.success(f"Predicted Player Value: €{predicted_value:,.2f}")

# Footer
st.markdown("---")
st.markdown("Created by group 6 with ❤️")