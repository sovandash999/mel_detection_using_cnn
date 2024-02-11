from PIL import Image
import streamlit as st
import io
import numpy as np
import tensorflow as tf
import keras

model=keras.models.load_model(r'C:\\Users\\sovan dash\\PycharmProjects\\melanoma_using_cnn\\meloma_cnn\\melanoma.h5')



def preprocess_uploaded_image(uploaded_file):
    try:
        img = Image.open(io.BytesIO(uploaded_file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255
        img_array = img_array.reshape((1, 224, 224, 3))
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

st.title('Skin Cancer Prediction')
class_names=['bening','malignant']
uploaded_img = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

st.subheader('this model has an accuracy of 91%')
c1,c2=st.columns(2)
if uploaded_img is not None:
    im_array = preprocess_uploaded_image(uploaded_img)

    with c1:
        st.image(uploaded_img, caption='Uploaded Image', use_column_width=True,width=40)

    with c2:
        if st.button('classify'):
            result=model.predict(im_array)[0,0]
            if result>=0.5:
                st.write('BENING')
                st.success(f'{result * 100:.2f}% chance of being bening')
            else:
                st.write('MALIGNANT')
                st.success(f'{(1-result) * 100:.2f}% chance of being malignant')
