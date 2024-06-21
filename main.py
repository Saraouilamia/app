import streamlit as st
import cv2
import tensorflow as tf
import lightgbm as lgb
import onnxruntime
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet
from torchvision import transforms

# Load models
mobilenet_model_path = r'C:\Users\HP\Potato_disease\models\MobileNetV2 - 16 juin\saved model\mobilenetv2_plant_disease.h5'
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)

lightgbm_model_path = r'C:/Users/HP/Potato_disease/models/lightgbm/lgb_model.txt'
lightgbm_model = lgb.Booster(model_file=lightgbm_model_path)

mobilenet_feature_model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

onnx_model_path = r'C:\Users\HP\Potato_disease\models\Google ViT\onnx infrence\plant_disease_vit (1).onnx'
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# Class names
class_names = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight',
    'Potato Late_blight', 'Potato healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
    'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato YellowLeaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# Preprocessing functions
def preprocess_image(img, model_type):
    if model_type in ['AgriNetBoost', 'MobileNetV2', 'Google ViT (ONNX)']:
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        if model_type == 'AgriNetBoost' or model_type == 'MobileNetV2':
            img_array = preprocess_input(img_array)
        if model_type == 'Google ViT (ONNX)':
            img_array = np.transpose(img_array, (0, 3, 1, 2)).astype(np.float32) / 255.0
        return img_array

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)

# Prediction functions
def predict_mobilenet(img_array):
    predictions = mobilenet_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

def extract_features(img_array):
    features = mobilenet_feature_model.predict(img_array)
    return features

def predict_lightgbm(img_array):
    features = extract_features(img_array)
    prediction = lightgbm_model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def predict_onnx(img_array):
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: img_array})
    probabilities = softmax(outputs[0][0])
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class] * 100
    return predicted_class, confidence

# Streamlit app
st.title("Plant Disease Detection")
model_type = st.selectbox("Choose a model", ["MobileNetV2", "AgriNetBoost", "Google ViT (ONNX)"])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    img_array = preprocess_image(img, model_type)

    if model_type == "MobileNetV2":
        predicted_class, confidence = predict_mobilenet(img_array)
    elif model_type == "AgriNetBoost":
        predicted_class, confidence = predict_lightgbm(img_array)
    elif model_type == "Google ViT (ONNX)":
        predicted_class, confidence = predict_onnx(img_array)

    prediction_text = f"Prediction: {class_names[predicted_class]}"
    if confidence is not None:
        prediction_text += f" with confidence {confidence:.2f}%"

    st.markdown(f"**<span style='color:#5B320C'>{prediction_text}</span>**", unsafe_allow_html=True)

# Camera functionality
st.write("Press 'Start Camera' to enable the camera feed and perform real-time predictions.")
camera = cv2.VideoCapture(0)

if st.button('Start Camera'):
    stframe = st.empty()
    stop_button_key = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to capture image")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels='RGB', use_column_width=True, caption='Camera Feed')

        img = Image.fromarray(rgb_frame)
        img_array = preprocess_image(img, model_type)

        if model_type == "MobileNetV2":
            predicted_class, confidence = predict_mobilenet(img_array)
        elif model_type == "AgriNetBoost":
            predicted_class, confidence = predict_lightgbm(img_array)
        elif model_type == "Google ViT (ONNX)":
            predicted_class, confidence = predict_onnx(img_array)

        prediction_text = f"Prediction: {class_names[predicted_class]}"
        if confidence is not None:
            prediction_text += f" with confidence {confidence:.2f}%"

        stframe.markdown(f"**<span style='color:#5B320C'>{prediction_text}</span>**", unsafe_allow_html=True)

        if st.button('Stop Camera', key=f"stop_camera_{stop_button_key}"):
            break
        stop_button_key += 1

camera.release()
