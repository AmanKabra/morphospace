import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('df.csv')

# Create a deep copy of the DataFrame for each case
df_copy = df.copy()

# Standardize 'performance' and 'tre' variables in df_copy1
scaler = StandardScaler()
df_copy[['performance', 'tre']] = scaler.fit_transform(df_copy[['performance', 'tre']])

# Melt both dataframes to long format for the ease of plotting
df_melt = df_copy.melt(id_vars=['density', 'size', 'checker_category', 'composition'], 
                  value_vars=['performance', 'tre'], 
                  var_name='outcome', 
                  value_name='value')

# Define the layout of the app
st.title("Faceted Scatter Plots")

# Faceted scatter plot for df_melt
fig = px.scatter(df_melt, 
                 x="density", 
                 y="value", 
                 color="checker_category", 
                 symbol="composition",  # Use different symbols for 'composition'
                 facet_row="outcome",
                 facet_col="size",
                 height=1200,  # Increase the height of the plot
                 title="Impact of Size, Density, Checker Category, and Composition on Performance and TRE")

st.plotly_chart(fig)