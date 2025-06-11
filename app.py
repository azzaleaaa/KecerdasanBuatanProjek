import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# --- PERBAIKAN 1: FUNGSI UNTUK MEMUAT MODEL DENGAN CACHING ---
# @st.cache_resource akan menyimpan output (model & processor) dalam cache.
# Fungsi ini hanya akan dijalankan SEKALI untuk setiap model_path.
# Pada pemanggilan berikutnya, hasilnya akan diambil dari cache secara instan.
@st.cache_resource
def load_model(model_path):
    st.info(f"Mengunduh dan memuat model: {model_path}...") # Pesan ini hanya muncul sekali
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    return processor, model

st.set_page_config(layout="wide") # Membuat layout lebih lebar
st.title("🔬 Perbandingan Model Deteksi Kanker Kulit")
st.markdown("Unggah gambar lesi kulit untuk dideteksi menggunakan dua model AI yang berbeda: Vision Transformer dan ConvNext.")


uploaded_file = st.file_uploader("Pilih gambar lesi kulit...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB") # Konversi ke RGB untuk konsistensi

    # --- PERBAIKAN 2: TATA LETAK MENGGUNAKAN KOLOM ---
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    with col2:
        st.subheader("Hasil Prediksi Model")
        
        models = {
            "Vision Transformer": "Anwarkh1/Skin_Cancer-Image_Classification",
            "ConvNext": "Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch"
        }

        for model_name, model_path in models.items():
            st.markdown(f"#### Model: **{model_name}**")
            with st.spinner(f"Memproses dengan {model_name}..."):
                try:
                    # Memanggil fungsi yang sudah di-cache
                    processor, model = load_model(model_path)
                    
                    # Proses inferensi tetap sama
                    inputs = processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    
                    pred_idx = torch.argmax(probs).item()
                    pred_class = model.config.id2label[pred_idx]
                    confidence = probs[0][pred_idx].item()
                    
                    # Menampilkan hasil
                    if confidence >= 0.5:
                        st.success(f"Prediksi: **{pred_class}**")
                        st.write(f"Tingkat Keyakinan: **{confidence:.2%}**")
                    else:
                        st.warning("⚠️ Keyakinan di bawah 50%. Mungkin gambar tidak sesuai.")
                
                # --- PERBAIKAN 3: PENANGANAN ERROR ---
                # Menangkap error jika model gagal diunduh atau ada masalah lain
                except Exception as e:
                    st.error(f"Gagal memproses dengan {model_name}. Error: {e}")
            st.markdown("---")


# Credit di luar, bisa diletakkan di sidebar atau di bawah
st.sidebar.markdown("""
---
### 🧠 Credit

**📦 Model:**
- [Vision Transformer by Anwarkh1](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)  
- [ConvNext by Pranavkpba2000](https://huggingface.co/Pranavkpba2000/convnext-fine-tuned-complete-skin-cancer-50epoch)

**🤖 Chat Assistant:**
- ChatGPT by OpenAI  
- Gemini by Google
""")
