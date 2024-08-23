import os
import pickle
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from data.load_220k_GPT4 import Load_Data

DATA_DIR = 'data'
WORKING_DIR = 'model'

class Extract_Image_Feature:
    def __init__(self, model = VGG16()):
        self.model = model
        self.features = None
    
    def Restructure_Model(self):
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

    def Summarize_Model(self):
        print(self.model.summary())

    def Extract_Img_Feature(self):
        self.Restructure_Model()
        load_data = Load_Data()
        df = load_data.Get_1000()

        features = {}

        for url in df['url']:
            # Đọc ảnh bằng url 
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            # get image ID
            image_id = url

            # store feature
            features[image_id] = self.get_custom_feature(img)
            print(features[image_id].shape)
        return features

    def Store_Features(self, file_name):
        features = self.Extract_Img_Feature()

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
    
    # Get feature:
    def get_custom_feature(self, img):
        # load vgg16 model
        model_VGG16 = self.model

        # Chuyển đổi ảnh thành RGB nếu cần
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize hình ảnh
        image = img.resize((224, 224))

        # convert image pixels to numpy array
        image = np.array(image)

        # Đảm bảo kích thước đúng định dạng (1, 224, 224, 3)
        image = image.reshape((1, 224, 224, 3))

        # preprocess image for vgg
        image = preprocess_input(image)

        # extract features
        feature = model_VGG16.predict(image, verbose=0)
        return feature