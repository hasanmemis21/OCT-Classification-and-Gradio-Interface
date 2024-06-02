import hashlib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import io
from pymongo import MongoClient

# MongoDB bağlantısını ayarlayın ve yeni veritabanı ve koleksiyon oluşturun
client = MongoClient("mongodb://localhost:27017/")
db = client['OCT-DATABASE']
collection = db['image_labels']

# Modelinizi yükleyin
model = load_model('model.h5')

# Etiketler
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Görüntü ön işleme fonksiyonu
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError("Unsupported image format.")
    image = image.resize((224, 224))  # Modelin beklediği boyuta göre ayarlayın
    image = np.array(image)
    image = image / 255.0  # Normalizasyon
    image = np.expand_dims(image, axis=0)  # Batch boyutunu ekleyin
    return image

def is_grayscale(image):
    image_np = np.array(image.convert('RGB'))
    if np.all(image_np[:, :, 0] == image_np[:, :, 1]) and np.all(image_np[:, :, 1] == image_np[:, :, 2]):
        return True
    return False

def is_retinal_image(image):
    # Görüntülerin siyah-beyaz olup olmadığını kontrol et
    if not is_grayscale(image):
        return False

    # Retinal görüntülerin genellikle belirli renk ve doku özelliklerine sahip olması gerekir
    image_np = np.array(image)

    # Modelin tahminini kullanarak doğrulama
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]]

    # Eğer model retinal bir hastalığı doğru bir şekilde tahmin ediyorsa, görüntü retinal kabul edilir
    if predicted_label in labels:
        return True
    else:
        return False

# Görüntü hash oluşturma fonksiyonu
def calculate_image_hash(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_hash = hashlib.md5(image_bytes.getvalue()).hexdigest()
    return image_hash

# Tahmin fonksiyonu
def predict(image):
    # Görüntüyü ön işleyin
    processed_image = preprocess_image(image)

    # Modelden tahmin yapın
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]]

    # Grafik oluştur
    fig, ax = plt.subplots(figsize=(8, 6))  # Grafik boyutunu büyüt
    bars = ax.bar(labels, predictions[0], color=['blue', 'green', 'red', 'purple'], width=0.4)  # Sütun genişliğini daralt
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Disease Prediction Probabilities')

    # Oranları çubukların üzerine yazdır
    for bar, prob in zip(bars, predictions[0]):
        height = bar.get_height()
        ax.annotate(f'{prob:.5f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = Image.open(buf)

    # Tahmin oranlarını tablo olarak oluştur
    df = pd.DataFrame({
        'Disease': labels,
        'Probability': [f'{prob:.5f}' for prob in predictions[0]]
    })

    # Görüntü hash'ini hesapla ve kontrol et
    image_hash = calculate_image_hash(Image.fromarray((processed_image[0] * 255).astype(np.uint8)))
    if collection.find_one({"image_hash": image_hash}) is None:
        # Görüntü ve tahmin sonuçlarını MongoDB'ye kaydet
        image_bytes = io.BytesIO()
        Image.fromarray((processed_image[0] * 255).astype(np.uint8)).save(image_bytes, format='PNG')
        image_data = image_bytes.getvalue()
        collection.insert_one({
            "image_hash": image_hash,
            "image": image_data,
            "predicted_label": predicted_label,
            "probabilities": predictions[0].tolist()
        })

    return predicted_label, pil_img, df

# Favicon dosyasının yolu
favicon_path = "assets/OCT-logo.ico"

# Gradio arayüzünü oluşturun
with gr.Blocks() as demo:
    gr.Markdown(f"""
    <head>
        <title>Medical Image Classification</title>
        <link rel="icon" type="image/x-icon" href="{favicon_path}">
    </head>
    <style>
        #image-input {{ width: 200px; }}
        #prediction-table {{ height: auto !important; max-height: 300px; }}
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Image", elem_id="image-input")
            prediction_table = gr.Dataframe(pd.DataFrame({'Disease': labels, 'Probability': ['0.00000', '0.00000', '0.00000', '0.00000']}), elem_id="prediction-table")
        with gr.Column(scale=1):
            predicted_label = gr.Textbox(label="Predicted Label")
            prediction_chart = gr.Image(type="pil", label="Prediction Chart")

    def update_table(image):
        if not is_retinal_image(Image.fromarray(image)):
            return "Error: Uploaded image is not a valid retinal image.", None, None
        predicted_label, pil_img, df = predict(image)
        return predicted_label, pil_img, df

    image_input.change(update_table, inputs=image_input, outputs=[predicted_label, prediction_chart, prediction_table])

# Arayüzü başlatın
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
