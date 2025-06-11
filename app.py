import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import torch

st.title("Perbandingan Model Deteksi Kanker Kulit")

# Identitas pembuat
st.markdown("""
**Aplikasi ini dibuat oleh:**  
**Nama:** Muhammad Firda Satria  
**NIM:** 2304130057  
**Prodi:** Teknik Informatika, Universitas Negeri Semarang 
""")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar lesi kulit", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Daftar model dan path lokal konfigurasinya
    models = {
        "Vision Transformer": {
            "model_path": "Anwarkh1/Skin_Cancer-Image_Classification",
            "local_config_path": "Model_Configs/Model Vit/"
        },
        "ConvNext": {
            "model_path": "Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch",
            "local_config_path": "Model_Configs/Model ConvNext/"
        }
    }

    for model_name, model_info in models.items():
        st.subheader(f"Model: {model_name}")
        with st.spinner(f"Memproses dengan {model_name}..."):
            # Load processor dan config dari lokal
            processor = AutoImageProcessor.from_pretrained(model_info["local_config_path"])
            config = AutoConfig.from_pretrained(model_info["local_config_path"])

            # Load model dari HuggingFace, tapi pakai config lokal
            model = AutoModelForImageClassification.from_pretrained(
                model_info["model_path"],
                config=config
            )

            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            pred_idx = torch.argmax(probs).item()
            pred_class = model.config.id2label[str(pred_idx)]
            confidence = probs[0][pred_idx].item()

            if confidence >= 0.5:
                st.write(f"Prediksi: **{pred_class}**")
                st.write(f"Akurasi Prediksi: **{confidence:.2%}**")
            else:
                st.write("⚠️ Akurasi yang dihasilkan dibawah 50%, silahkan menggunakan gambar yang sesuai")

# Credit
st.markdown("""
---
### 🧠 Credit

**📦 Model:**
- [Vision Transformer by Anwarkh1](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)
- [ConvNext by Pranavkpba2000](https://huggingface.co/Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch)

**🤖 Chat Assistant:**
- ChatGPT by OpenAI
- Gemini by Google
""")
