import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.interpolate import griddata

df = pd.read_csv('df_composition.csv')

checker_options = ["One", "All"]
density_options = [0, 0.4, 1]

checker_input = st.selectbox("Select Checker Category:", checker_options)
density_input = st.selectbox("Select Density Category:", density_options)

df = df[(df['checker_category'] == checker_input) & (df['density_category'] == density_input)]

axes_options = ["index_expertise_level", "index_innovation_level", "index_doing_level", "index_linking_level", "index_challenging_level"]

x_axis = st.selectbox("Select X-axis:", axes_options)
y_axis = st.selectbox("Select Y-axis:", axes_options)

sizes = [3, 6, 9, 12]

def prepare_interpolated_figure(df, sizes, z_var):
    fig = sp.make_subplots(rows=1, cols=4, subplot_titles=[f'Size {size}' for size in sizes], specs=[[{'type': 'scatter3d'}]*4])

    for idx, size in enumerate(sizes):
        df_size = df[df["size"] == size]
        scatter = px.scatter_3d(df_size, x=x_axis, y=y_axis, z=z_var, color=z_var)
        fig.add_trace(scatter.data[0], row=1, col=idx+1)

        x = df_size[x_axis]
        y = df_size[y_axis]
        z = df_size[z_var]
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

        surface = go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Viridis', showscale=False)
        fig.add_trace(surface, row=1, col=idx+1)

    fig.update_layout(height=600, width=1200, title_text="Subplots of different sizes with surface",
                      scene = dict(xaxis_title=x_axis,
                                   yaxis_title=y_axis,
                                   zaxis_title=z_var),
                      scene2 = dict(xaxis_title=x_axis,
                                    yaxis_title=y_axis,
                                    zaxis_title=z_var),
                      scene3 = dict(xaxis_title=x_axis,
                                    yaxis_title=y_axis,
                                    zaxis_title=z_var),
                      scene4 = dict(xaxis_title=x_axis,
                                    yaxis_title=y_axis,
                                    zaxis_title=z_var))

    return fig

def prepare_figure(df, sizes, z_var, degree):
    fig = sp.make_subplots(rows=1, cols=4, subplot_titles=[f'Size {size} - {z_var}' for size in sizes], specs=[[{'type': 'scatter3d'}]*4])

    for idx, size in enumerate(sizes):
        df_size = df[df["size"] == size]
        scatter = px.scatter_3d(df_size, x=x_axis, y=y_axis, z=z_var, color=z_var)
        fig.add_trace(scatter.data[0], row=1, col=idx+1)

        poly = PolynomialFeatures(degree=degree)
        X = df_size[[x_axis, y_axis]]
        y = df_size[z_var]

        X_poly = poly.fit_transform(X)

        model = LinearRegression().fit(X_poly, y)

        x_grid, y_grid = np.meshgrid(np.linspace(df_size[x_axis].min(), df_size[x_axis].max(), num=10),
                                     np.linspace(df_size[y_axis].min(), df_size[y_axis].max(), num=10))

        grid_df = pd.DataFrame(np.vstack([x_grid.ravel(), y_grid.ravel()]).T, columns=X.columns)

        z_grid = model.predict(poly.transform(grid_df)).reshape(x_grid.shape)

        surface = go.Surface(x=x_grid, y=y_grid, z=z_grid, colorscale='Viridis', showscale=False)
        fig.add_trace(surface, row=1, col=idx+1)

    fig.update_layout(height=600, width=1200, title_text=f"Subplots of different sizes with polynomial regression surface for {z_var}",
                      scene = dict(xaxis_title=x_axis,
                                   yaxis_title=y_axis,
                                   zaxis_title=z_var),
                      scene2 = dict(xaxis_title=x_axis,
                                    yaxis_title=y_axis,
                                    zaxis_title=z_var),
                      scene3 = dict(xaxis_title=x_axis,
                                    yaxis_title=y_axis,
                                    zaxis_title=z_var),
                      scene4 = dict(xaxis_title=x_axis,
                                    yaxis_title=y_axis,
                                    zaxis_title=z_var))

    return fig

st.title('Composition variations')

st.header("First, we will see polynomial regression fits.")
degree = st.slider('Choose in this slider the polynomial degree for the regression plot:', 
                   min_value=1, 
                   max_value=5, 
                   value=3, 
                   step=1)

fig1 = prepare_figure(df, sizes, 'performance', degree)
fig2 = prepare_figure(df, sizes, 'tre', degree)

st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.header("The following are surface fitness plots - peaks and valleys - that Federico has been wanting to see.")
st.write("I don't know how to make analytical sense of these.")

fig3 = prepare_interpolated_figure(df, sizes, 'performance')
fig4 = prepare_interpolated_figure(df, sizes, 'tre')

st.plotly_chart(fig3)
st.plotly_chart(fig4)