import streamlit as st
import pickle
import numpy as np



model = pickle.load(open('C:/Users/DELL/Desktop/Fake_bills_prediction_app/random_forest_classifier.pkl','rb'))



def predict_genuine_fake(diagonal,height_left,height_right,margin_low,margin_up,length):
    input = np.array([[diagonal,height_left,height_right,margin_low,margin_up,length]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)


def main():
    st.title("Genuine or Fake Bill Prediction")
    html_temp = """
    <div style="background:black; padding:10px; ">
    <h2 style="color:white;text-align:center;"> Abalone Age Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)