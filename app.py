from data_processing import process_datasets
from plots import (
    stacked_barchart,
    donut_chart,
    stacked_area_chart,
    treemap,
    waterfall_chart,
    buble_chart,
    series_wordcloud,
    movies_wordcloud
)

import pandas as pd
import streamlit as st
from PIL import Image
import os


# conda deactivate
# source .pytradex_venv/bin/activate
# streamlit run app.py
if __name__ == '__main__':
    # Set Page Config
    st.set_page_config(layout="wide")

    # Write logo
    col0, col1, col2 = st.columns([5, 2.5, 5])
    col1.image(Image.open(os.path.join("resources", "netflix_logo.png")))

    # Write Title
    col0, col1, col2 = st.columns([3, 8, 3])
    col1.markdown(
        '<p style="font-family:sans-serif; color:#EB2828; font-size: 65px; font-weight: bold; text-align: center;"'
        '>Watch History Analysis</p>',
        unsafe_allow_html=True
    )

    # Add empty space
    st.write("#")

    # Write a line
    st.write("-----")

    # Load dataset
    df: pd.DataFrame = process_datasets()

    # Plot Stacked barchart
    col0, col1, col2, col3, col4 = st.columns([1, 8, 1, 8, 0.5])
    fig = stacked_barchart(df=df.copy())
    col1.plotly_chart(fig, use_container_width=True)

    # Plot donut chart
    fig = donut_chart(df=df.copy())
    col3.plotly_chart(fig, use_container_width=True)

    # Add empty space
    st.write("#")

    # Plot buble chart
    col0, col1, col2 = st.columns([3, 4, 3])
    fig = buble_chart(df=df.copy())
    col1.pyplot(fig)

    # Add empty space
    st.write("#")

    # Plot stacked area charts
    col0, col1, col2, col3, col4 = st.columns([0.5, 8, 1, 8, 0.25])
    fig = stacked_area_chart(df=df.copy())
    col1.plotly_chart(fig, use_container_width=True)

    # Plot Waterfall charts
    fig = waterfall_chart(df=df.copy())
    col3.plotly_chart(fig, use_container_width=True)

    # Add empty space
    st.write("#")

    # Plot treemsp
    col0, col1, col2 = st.columns([1, 8, 1])
    fig = treemap(df=df.copy(), others_n=15)
    col1.plotly_chart(fig, use_container_width=True)

    # Add empty space
    st.write("#")

    # Plot Stacked barchart
    col0, col1, col2, col3, col4 = st.columns([1, 8, 1, 8, 0.5])
    fig = series_wordcloud(df=df.copy())
    col1.plotly_chart(fig, use_container_width=True)

    # Plot donut chart
    fig = movies_wordcloud(df=df.copy())
    col3.plotly_chart(fig, use_container_width=True)
    


