import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Player Comparison Dashboard", page_icon="🔍", layout="wide")

st.title("Player Comparison Dashboard")

st.markdown("Navigate to: [Main Dashboard](http://localhost:8501)")

fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    os.chdir(r"C:\Users\aashi\Downloads\archive")
    df = pd.read_csv("data_22_23_4.csv", encoding = "ISO-8859-1")
    
# Assuming you have a DataFrame named 'df' containing your FIFA player data

# Group the data by "short_name" and calculate the average overall rating
averaged_df = df.groupby('short_name')['overall'].mean().reset_index()

# Calculate average overall rating
average_overall = averaged_df['overall'].mean()

# Count the number of unique players
player_count = averaged_df.shape[0]

# Create a histogram showing the distribution of overall ratings
fig1 = px.histogram(averaged_df, x="overall", title="Distribution of Overall Player Ratings")
st.plotly_chart(fig1)

# Display the average overall rating and the count of players
st.write(f"Average Overall Rating: {average_overall:.2f}")
st.write(f"Number of Unique Players: {player_count}")


# Group the data by "short_name" directly and calculate the average overall rating
averaged_df = df.groupby('short_name')['overall'].mean().reset_index()

# Filter for players with an average overall rating above 80
filtered_df = averaged_df[averaged_df['overall'] >= 80]

# Sort the filtered data in descending order by average overall rating
sorted_df = filtered_df.nlargest(20, 'overall')

# Bar Chart of Top 20 Players (starting from 80)
fig = px.bar(sorted_df, x='short_name', y='overall', title="Top 20 Players (Average Overall, Starting from 80)")
fig.update_layout(yaxis=dict(range=[80, None]))  # Set the y-axis range to start at 80
st.plotly_chart(fig)

# Assuming you have a DataFrame named 'df' containing your FIFA player data

selected_players = st.multiselect("Select Players", df['short_name'].unique())

if selected_players:
    filtered_df = df[df['short_name'].isin(selected_players)]

    # Filter for years 22 and 23
    filtered_df = filtered_df[filtered_df['fifa_version'].isin([22, 23])]

    # Create a scatter plot
    fig = px.scatter(
        filtered_df,
        x='fifa_version',
        y='overall',
        color='short_name',
        title=f"Player Growth from 22 to 23",
        trendline='ols'  # Add a linear regression line
    )

    st.plotly_chart(fig)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have a DataFrame named 'df' containing your FIFA player data

# Select features and target variable
X = df[['age', 'height_cm', 'weight_kg', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]
y = df['overall']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.write("Model Evaluation:")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Create a table of actual vs. predicted overall ratings
results_df = pd.DataFrame({'Actual Overall': y_test, 'Predicted Overall': y_pred})

# Display the table as a scrollable dataframe, showing 10 entries at a time
st.write("Actual vs. Predicted Overall Ratings:")
st.dataframe(results_df, height=200)  # Adjust height to allow 10 rows and scrolling

# Display the top 5 features that contribute most to overall rating
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
top_features = feature_importances.nlargest(5, 'Importance')
st.write("Top Features Contributing to Overall Rating:")
st.dataframe(top_features)

# Use the trained model to make predictions for new players
new_player_data = pd.DataFrame({'age': [25], 'height_cm': [180], 'weight_kg': [70],
                                'pace': [85], 'shooting': [90], 'passing': [80],
                                'dribbling': [85], 'defending': [80], 'physic': [80]})

predicted_overall = model.predict(new_player_data)
st.write(f"Predicted Overall Rating for New Player: {predicted_overall[0]:.2f}")

# Visualize the relationship between age and overall rating
age_corr_df = df[['age', 'overall']]
fig = px.scatter(age_corr_df, x='age', y='overall', title="Relationship Between Age and Overall Rating")
st.plotly_chart(fig)

# Visualize the relationship between height and overall rating
height_corr_df = df[['height_cm', 'overall']]
fig = px.scatter(height_corr_df, x='height_cm', y='overall', title="Relationship Between Height and Overall Rating")
st.plotly_chart(fig)

####################################################################################################################################################################

# Player selection
players = st.multiselect("Select players to compare", df["short_name"].unique(), max_selections=5)

if players:
    player_data = df[df["short_name"].isin(players)]
    
    # Radar chart
    attributes = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
    fig = px.line_polar(player_data, r=attributes, theta=attributes, line_close=True, 
                        color="short_name", hover_name="short_name", range_r=[0,100])
    st.plotly_chart(fig)
    
    # Detailed comparison table
    st.write(player_data[["short_name", "overall", "potential", "value_eur", "wage_eur"] + attributes])

else:
    st.write("Please select players to compare.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by group 6 specially for Harnal Sir and Amarnath Sir using Streamlit")