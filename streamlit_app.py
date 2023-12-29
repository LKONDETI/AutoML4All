import streamlit as st
import pandas as pd 
import os

# Import profiling capability
# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#ML Stuff
from pycaret.regression import setup, compare_models, pull, save_model
# from classifi import run_classification_program

with st.sidebar:
    st.image("https://img.freepik.com/free-photo/cardano-blockchain-platform_23-2150411956.jpg")
    st.title("AutoStreamML")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application allows you to bulid an automated ML pipelince using Streamlit, Pandas Profiling and PyCaret.")

if choice == "Upload":
    st.title("Upload Your Dataset for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df= pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv", index_col=None)


if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    report = df.profile_report()
    st_profile_report(report)

if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your target", df.columns)
    st.session_state.selected_target = target
    model = st.radio("Navigation",["Convert Regression Model","Convert classification Model"])
    if model == "Convert Regression Model":
        st.title("Regression Model")
        setup(df, target= target)
        setup_df = pull()
        st.info("This is the ML experiment settings")
        st.dataframe(setup_df)
        best_model= compare_models()
        compare_df= pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')

    # if model =="Convert classification Model":
    #     st.title("Classification Model")
    #     run_classification_program()  

    

        

if choice == "Download":
    pass