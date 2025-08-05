import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title("My First Streamlit App")

# display a simple text
st.write("Hello, world!")
df= pd.DataFrame({
    "Column 1": [1, 2, 3, 4, 5],
    "Column 2": [10, 20, 30, 40, 50]})

# display the dataframe
st.write("Here is a simple dataframe:")
st.dataframe(df)


# create a line chart
st.write("Here is a simple line chart:")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"]
)
st.line_chart(chart_data)
