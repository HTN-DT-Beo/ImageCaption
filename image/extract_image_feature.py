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
        load_data = Load_Data()
        df = load_data.Get_1000()
        print(df.head(10))

        features = {}

        for url in df['url']:
            # Đọc ảnh bằng url 
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            # Chuyển đổi ảnh thành RGB nếu cần
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize hình ảnh
            image = img.resize((224, 224))

            # convert image pixels to numpy array
            image = np.array(image)
            print(image.shape)

            # Đảm bảo kích thước đúng định dạng (1, 224, 224, 3)
            image = image.reshape((1, 224, 224, 3))
            print(image.shape)

            # preprocess image for vgg
            image = preprocess_input(image)

            # extract features
            feature = self.model.predict(image, verbose=0)

            # get image ID
            image_id = url

            # store feature
            features[image_id] = feature
        return features
    
    def Store_Features(self):
        features = self.Extract_Img_Feature()

        # store features in pickle
        file_path = os.path.join(WORKING_DIR, '220k_GPT4_features_top1000.pkl')

        # Tạo các thư mục nếu chúng chưa tồn tại
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Lưu dữ liệu vào tệp
        with open(file_path, 'wb') as f:
            pickle.dump(features, f)

    def Load_Features(self):
        with open(os.path.join(WORKING_DIR, '220k_GPT4_features_top1000.pkl'), 'rb') as f:
            features = pickle.load(f)
        return features