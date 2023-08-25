import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.subplots as sp
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('df_aggregated.csv')

def prepare_interpolated_figure(df, z_var, density):
    df = df[df['density'] == density]
    fig = sp.make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    scatter = px.scatter_3d(df, x='index_expertise_level', y='index_innovation_level', z=z_var, color=z_var)
    fig.add_trace(scatter.data[0])

    grouped = df.groupby(['index_expertise_level', 'index_innovation_level']).agg({z_var: 'mean'}).reset_index()
    x = grouped['index_expertise_level']
    y = grouped['index_innovation_level']
    z = grouped[z_var]
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    surface = go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Viridis', showscale=False)
    fig.add_trace(surface)
    fig.update_layout(height=600, width=600, title_text=f"3D Plot with Surface (Density = {density})",
                      scene=dict(xaxis_title='Expertise Level', yaxis_title='Innovation Level', zaxis_title=z_var))
    
    return fig

df_all = df[df['checker_category'] == 'All']
df_one = df[df['checker_category'] == 'One']

densities = [0, 0.4, 1]
z_vars = ["performance", "tre"]

def display_plots_for_z_var(df, z_var_name):
    st.subheader(f"{z_var_name.capitalize()}")
    cols = st.columns(len(densities)) # Create columns based on the number of densities
    for i, density in enumerate(densities):
        fig = prepare_interpolated_figure(df, z_var_name, density)
        cols[i].plotly_chart(fig)

st.title("Fitness Plots")

st.header("Performance Plots")
st.subheader("For All")
display_plots_for_z_var(df_all, "performance")

st.subheader("For One")
display_plots_for_z_var(df_one, "performance")

st.header("Tre Plots")
st.subheader("For All")
display_plots_for_z_var(df_all, "tre")

st.subheader("For One")
display_plots_for_z_var(df_one, "tre")