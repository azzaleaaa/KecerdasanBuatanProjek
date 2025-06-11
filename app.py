import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Judul aplikasi
st.title("Perbandingan Model Deteksi Kanker Kulit")

# Identitas pembuat
st.markdown("""
**Aplikasi ini dibuat oleh:**  
👨‍💻 Muhammad Firda Satria  
📚 Mahasiswa Informatika, Universitas Negeri Semarang  
""")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar lesi kulit", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Nama model dari Hugging Face
    models = {
        "Vision Transformer": "Anwarkh1/Skin_Cancer-Image_Classification",
        "ConvNeXT": "Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch" 
    }

    for model_name, hf_model_id in models.items():
        st.subheader(f"Model: {model_name}")
        with st.spinner(f"Memproses dengan {model_name}..."):
            processor = AutoImageProcessor.from_pretrained(hf_model_id)
            model = AutoModelForImageClassification.from_pretrained(hf_model_id)

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
                st.write("⚠️ Model tidak cukup yakin untuk melakukan prediksi (akurasi < 50%).")

# Credit
st.markdown("""
---
🧠 Model berbasis Hugging Face Transformers  
📦 Powered by [🤗 Hugging Face](https://huggingface.co) dan [Streamlit](https://streamlit.io)  
""")
