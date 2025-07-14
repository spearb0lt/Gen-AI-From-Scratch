import streamlit as st
import pandas as pd

st.title("Streamlit Widgets Example")
name =  st.text_input("Enter your name:")
# age = st.number_input("Enter your age:", min_value=0, max_value=120
if name:
    st.write(f"Hello, {name}!")


age= st.select_slider("Select your age:", options=list(range(0, 121)), value=30)
st.write(f"You selected age: {age}")

options=["Option 1", "Option 2", "Option 3"
         ]
selected_option = st.selectbox("Choose an option:", options)
st.write(f"You selected: {selected_option}")





data={
    "Column 1": [1, 2, 3, 4, 5],
    "Column 2": [10, 20, 30, 40, 50],
    "Column 3": [100, 200, 300, 400, 500]
}

df= pd.DataFrame(data)
df.to_csv("data.csv", index=False)
st.write(df)




upload_file = st.file_uploader("Upload a CSV file", type=["csv"])
if upload_file is not None:
    df = pd.read_csv(upload_file)





