import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

st.title("Perbandingan Model Deteksi Kanker Kulit")

# Identitas pembuat
st.markdown("""
**Aplikasi ini dibuat oleh:**  
Nama : Muhammad Firda Satria  
NIM : 2304130057
Prodi : Teknik Informatika, Universitas Negeri Semarang  
""")

uploaded_file = st.file_uploader("Unggah gambar lesi kulit", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    models = {
        "Vision Transformer": "Anwarkh1/Skin_Cancer-Image_Classification",
        "ConvNext": "Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch"
    }

    for model_name, model_path in models.items():
        st.subheader(f"Model: {model_name}")
        with st.spinner(f"Memproses dengan {model_name}..."):
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            pred_idx = torch.argmax(probs).item()
            pred_class = model.config.id2label[pred_idx]
            confidence = probs[0][pred_idx].item()

            if confidence >= 0.5:
                st.write(f"Prediksi: **{pred_class}**")
                st.write(f"Akurasi Prediksi: **{confidence:.2%}**")
            else:
                st.write("⚠️ Akurasi yang dihasilkan dibawah 50%, silahkan menggunakan gambar yang sesuai")

# Credit
st.markdown("""
---
🧠 Credit  
📦 [Model Vision Transformer By Anwarkh1](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification) 
📦 [Model ConvNext By Pranavkpba2000](https://huggingface.co/Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch)  
""")
