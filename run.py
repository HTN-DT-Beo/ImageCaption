from train.train import ModelTrainer
from predict import predict
from image import extract_image_feature

import streamlit as st
from PIL import Image

DATA_DIR = 'data'
MODEL_DIR = 'LSTM_2wordGC'
EXTRACT_IMAGE_FEATURE_FILE = '220k_GPT4_features_.pkl'

# --------------------------- EXTRACT IMAGE FEATURE ------------------------

# Prepare Image Feature
# eif = extract_image_feature.Extract_Image_Feature()
# eif.Store_Features(EXTRACT_IMAGE_FEATURE_FILE)


# --------------------------- TRAINING --------------------------

trainer = ModelTrainer()
trainer.train_model(MODEL_DIR)
features, mapping, tokenizer, vocab_size, max_length, train = trainer.prepare_data()

'''
# --------------------------- IMAGE -----------------------

# Tạo tiêu đề cho ứng dụng
st.title("IMAGE CAPTION")

# Sử dụng streamlit để tải lên hình ảnh
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
image = None
# Kiểm tra nếu có file được tải lên
if uploaded_file is not None:
    # Mở hình ảnh bằng PIL
    image = Image.open(uploaded_file)
    
    # Hiển thị hình ảnh trên giao diện Streamlit
    st.image(image, caption='Uploaded Image.', use_column_width=True)


# --------------------------- LOAD MODEL ------------------------

# Load Wieghts
model = trainer.load_weights(MODEL_DIR, vocab_size, max_length)


# ---------------------- CAPTION --------------------

if st.button("Generate Caption"):
    caption = predict.generate_predict_caption(image, model, tokenizer, max_length)
    st.write("Image Caption:", caption)

'''