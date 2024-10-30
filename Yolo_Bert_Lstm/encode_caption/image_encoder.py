import os
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from PIL import Image
import requests
from io import BytesIO

import tensorflow
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Flatten

from transformers import BertTokenizer, BertModel
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.tasks import attempt_load_one_weight

import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import cv2
from torchvision import transforms
from ultralytics import YOLO


WORKING_DIR = './'

# Hàm tiền xử lý ảnh thành tensor cho YOLO
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # Chuyển từ numpy sang PIL
        transforms.Resize((640, 640)),  # Thay đổi kích thước ảnh về 640x640
        transforms.ToTensor(),  # Chuyển thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo chuẩn của ImageNet
    ])
    tensor_image = preprocess(image).unsqueeze(0)  # Thêm batch dimension
    return tensor_image

def get_backbone_model():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    # Access the backbone layers
    backbone = model.model.model[:10]  # Layers 0 to 9 form the backbone
    # Create a new Sequential model with just the backbone layers
    backbone_model = torch.nn.Sequential(*backbone)
    return backbone_model

def save_backbone_model():
    backbone_model = get_backbone_model()
    # Save the backbone model
    torch.save(backbone_model.state_dict(), 'yolov8n_backbone.pt')

# def load_backbone_model(backbone):
#     # Load the model
#     backbone_model = torch.nn.Sequential(*backbone)
#     backbone_model.load_state_dict(torch.load('yolov8n_backbone.pt'))
def load_backbone_model():
    # Load the model
    backbone_model = torch.nn.Sequential()
    backbone_model.load_state_dict(torch.load('yolov8n_backbone.pt'))

# If you want to use the backbone for feature extraction
def extract_features(backbone_model, image):
    with torch.no_grad():
        features = backbone_model(image)
    return features

def extract_features_data(df):
    # load backbone_model
    backbone_model = load_backbone_model()

    # extract features from image
    features = {}

    for url in tqdm(df['url']):
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # Đọc ảnh bằng OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Chuyển đổi từ BGR (OpenCV) sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Chuyển đổi ảnh thành định dạng tensor phù hợp cho mô hình
        transformed_image = preprocess_image(image_rgb)
        feature_matrix = extract_features(backbone_model, transformed_image)

        # get image ID
        image_id = url

        # store feature
        features[image_id] = feature_matrix
    return features

def Store_Features(self, file_name, features):
    # store features in pickle
    file_path = os.path.join(WORKING_DIR, file_name)

    # Tạo các thư mục nếu chúng chưa tồn tại
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Lưu dữ liệu vào tệp
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)

def Load_Features(self, file_name):
    with open(os.path.join(WORKING_DIR, file_name), 'rb') as f:
        features = pickle.load(f)
    return features

def run(df):
    # Extract --> Save Features --> Load Features
    features_file_name = '220k_GPT4_features_8k.pkl'
    Store_Features(features_file_name, extract_features_data(df))
    return Load_Features(features_file_name)

# Test
data = pd.read_parquet("hf://datasets/laion/220k-GPT4Vision-captions-from-LIVIS/lvis_caption_url.parquet")
df = data.head(2)
run(df)


