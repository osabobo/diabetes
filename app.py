import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
cv_model = open('predicts_dia.pkl', 'rb')
cv = joblib.load(cv_model)

def prediction(Pregnancies,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,Diabetes_Pedigree_Function,Age):
    Pregnancies=Pregnancies


    Glucose=Glucose
    Blood_Pressure=Blood_Pressure
    Skin_Thickness=Skin_Thickness
    Insulin=Insulin
    BMI=BMI
    Diabetes_Pedigree_Function=Diabetes_Pedigree_Function

    Age=Age
     # Making predictions

    prediction = cv.predict(
        [[Pregnancies,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,Diabetes_Pedigree_Function,Age]])
    if prediction == 0:
        pred = 'No'
    else:
        pred = 'Yes'
    return pred

def main ():
    from PIL import Image
    image = Image.open('logo.jpg')
    image_spam = Image.open('images.jpg')
    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict whether patience has diabetes or not')


    st.sidebar.image(image_spam)





    st.title("Diabetes Prediction App")

    if add_selectbox == 'Online':
         Pregnancies=st.text_input('Pregnancies(Input numbers)')
         Glucose=st.text_input('Glucose(Input numbers)')
         Blood_Pressure=st.text_input('BloodPressure(Input numbers)')
         Skin_Thickness=st.text_input('SkinThickness(Input numbers)')
         Insulin=st.text_input('Insulin(Input numbers)')
         BMI=st.text_input('BMI(Input numbers)')
         Diabetes_Pedigree_Function=st.text_input('DiabetesPedigreeFunction(Input numbers)')


         Age = st.text_input('Age')



         result=""




         if st.button("Predict"):
             result = prediction(Pregnancies,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,Diabetes_Pedigree_Function,Age)
             st.success(result)








    if add_selectbox == 'Batch':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        file_upload = st.file_uploader("Upload csv file for predictions", type="csv")





        st.title('Make sure the csv File is in the same format  as diabetex.csv before uploading to avoid Error')

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data=data.drop('Outcome', axis=1)


            predictions = cv.predict(data)





            st.write(predictions)



if __name__ == '__main__':
    main()
