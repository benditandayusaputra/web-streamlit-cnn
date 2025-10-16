import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import io
import os

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

def load_old_model_weights(model_path):
    import h5py
    import json
    
    with h5py.File(model_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            return None
        model_config = json.loads(model_config.decode('utf-8'))
        
        if 'config' in model_config and 'layers' in model_config['config']:
            for layer in model_config['config']['layers']:
                if 'batch_shape' in layer['config']:
                    batch_shape = layer['config']['batch_shape']
                    layer['config']['batch_input_shape'] = batch_shape
                    del layer['config']['batch_shape']
        
        model = keras.models.model_from_json(json.dumps(model_config))
        
        if 'model_weights' in f:
            weight_layer_names = [n.decode('utf8') for n in f['model_weights'].attrs['layer_names']]
            for layer_name in weight_layer_names:
                g = f['model_weights'][layer_name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                weights = [g[weight_name] for weight_name in weight_names]
                layer = model.get_layer(name=layer_name)
                layer.set_weights(weights)
        
        return model

@st.cache_resource
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        try:
            model = load_old_model_weights(model_path)
            if model is not None:
                st.warning("Model loaded dengan konversi format lama")
                return model
        except Exception as e2:
            pass
        
        st.error(f"Error loading model: {str(e)}")
        st.error("Model tidak kompatibel. Silakan convert model terlebih dahulu.")
        return None

def preprocess_image(_image, model):
    input_shape = model.input_shape
    if len(input_shape) == 4:
        height, width = input_shape[1], input_shape[2]
    else:
        height, width = 224, 224  

    target_size = (width, height)

    if _image.mode != "RGB":
        _image = _image.convert("RGB")

    img = _image.resize(target_size)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_with_confidence(model, processed_image, class_names):
    predictions = model.predict(processed_image, verbose=0)
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_probs = predictions[0][top_indices]
    
    results = []
    for idx, prob in zip(top_indices, top_probs):
        results.append({
            'class': class_names[idx],
            'confidence': float(prob * 100)
        })
    
    return results

def apply_image_augmentation(image):
    img_array = np.array(image)
    images = [img_array]
    images.append(cv2.flip(img_array, 1))
    
    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)
    
    for angle in [-10, 10]:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_array, matrix, (width, height))
        images.append(rotated)
    
    return images

st.title("🖼️ CNN Image Classifier")
st.markdown("---")

with st.sidebar:
    st.header("Konfigurasi")
    
    st.subheader("1. Model")
    model_file = st.file_uploader(
        "Upload Model (.h5) - Opsional",
        type=['h5'],
        help="Upload untuk mengganti model default"
    )
    
    st.subheader("2. Class Names")
    class_names_input = st.text_area(
        "Masukkan nama kelas (pisahkan dengan koma)",
        value="airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck",
        help="Contoh: cat,dog,bird"
    )
    
    st.subheader("3. Image Settings")
    img_size = st.selectbox(
        "Input Size",
        options=[32, 64, 224, 128, 256],
        index=0
    )
    
    use_tta = st.checkbox(
        "Gunakan Test Time Augmentation (TTA)",
        value=False,
        help="Meningkatkan akurasi dengan menguji beberapa variasi gambar"
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Gambar")
    uploaded_files = st.file_uploader(
        "Pilih gambar untuk diklasifikasi",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} gambar berhasil di-upload")

with col2:
    st.subheader("Hasil Prediksi")

default_model_path = "default_model.h5"

if uploaded_files:
    with st.spinner("Loading model..."):
        if model_file:
            model = load_model(model_file)
            st.info("Menggunakan model yang di-upload")
        elif os.path.exists(default_model_path):
            model = load_model(default_model_path)
            st.info("Menggunakan model default: default_model.h5")
        else:
            st.error(f"Model default '{default_model_path}' tidak ditemukan. Silakan upload model.")
            model = None
    
    if model is not None:
        class_names = [name.strip() for name in class_names_input.split(',')]
        st.success(f"Model loaded! Detected {len(class_names)} classes")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown("---")
            img_col, result_col = st.columns([1, 1])
            
            with img_col:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {idx+1}: {uploaded_file.name}", use_container_width=True)
            
            with result_col:
                with st.spinner(f"Memproses gambar {idx+1}..."):
                    if use_tta:
                        augmented_images = apply_image_augmentation(image)
                        all_predictions = []
                        
                        for aug_img in augmented_images:
                            aug_pil = Image.fromarray(aug_img.astype('uint8'))
                            processed = preprocess_image(aug_pil, model)
                            preds = model.predict(processed, verbose=0)
                            all_predictions.append(preds[0])
                        
                        avg_predictions = np.mean(all_predictions, axis=0)
                        top_indices = np.argsort(avg_predictions)[-3:][::-1]
                        results = []
                        for idx_pred, prob in zip(top_indices, avg_predictions[top_indices]):
                            results.append({
                                'class': class_names[idx_pred],
                                'confidence': float(prob * 100)
                            })
                    else:
                        processed_image = preprocess_image(image, model)
                        results = predict_with_confidence(model, processed_image, class_names)
                    
                    st.markdown("### Prediksi:")
                    
                    top_result = results[0]
                    if top_result['confidence'] > 80:
                        confidence_color = "🟢"
                    elif top_result['confidence'] > 50:
                        confidence_color = "🟡"
                    else:
                        confidence_color = "🔴"
                    
                    st.markdown(f"### {confidence_color} **{top_result['class'].upper()}**")
                    st.markdown(f"**Confidence: {top_result['confidence']:.2f}%**")
                    st.progress(top_result['confidence'] / 100)
                    
                    st.markdown("#### Top 3 Predictions:")
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. **{result['class']}**: {result['confidence']:.2f}%")
                        st.progress(result['confidence'] / 100)
else:
    st.info("ℹ️ Upload gambar untuk memulai klasifikasi")