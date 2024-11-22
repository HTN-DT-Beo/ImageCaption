import streamlit as st
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import requests
import numpy as np
from ultralytics import YOLO
import warnings

# Tắt cảnh báo Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Tải mô hình YOLOv8 backbone
model_yolo = YOLO('yolov8n.pt')
backbone = model_yolo.model.model[:10]  # Lấy backbone layers
backbone_model = torch.nn.Sequential(*backbone)


class FeatureExtractorModel(nn.Module):
    def __init__(self, backbone_model):
        super(FeatureExtractorModel, self).__init__()
        self.backbone_model = backbone_model
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        tensor_image = self.preprocess(image).unsqueeze(0)  # Thêm batch dimension
        return tensor_image

    def forward(self, image):
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            features = self.backbone_model(image_tensor)

        # Điều chỉnh đầu ra của features
        if features.shape[1] != 3 or features.shape[2:] != (224, 224):
            features = torch.nn.functional.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
            features = features[:, :3, :, :]  # Chỉ giữ 3 kênh đầu
        return features


# Đường dẫn đến mô hình
model_path = 'D:\Image_Caption\ImageCaption\Yolo_Gpt2\model'
model = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = FeatureExtractorModel(backbone_model)


# Hàm dự đoán caption
def predict_caption(model, image_url):
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = feature_extractor(image_rgb)

    output_ids = model.generate(pixel_values, max_length=150, min_length=10, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return image_rgb, caption


# Streamlit App
st.title("Image Captioning with YOLO-GPT2")

# Input URL
image_url = st.text_input("Enter the URL of the image:", "")

if image_url:
    try:
        # Display image and generate caption
        st.write("Processing the image...")
        image_rgb, caption = predict_caption(model, image_url)

        # Sử dụng hai cột
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Generated Caption:")
            st.write(caption)
    except Exception as e:
        st.error(f"An error occurred: {e}")
