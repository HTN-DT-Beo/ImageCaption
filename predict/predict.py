from image.extract_image_feature import Extract_Image_Feature as EIF
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import numpy as np
import requests


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index  with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach and tag
        if word == 'endseq':
            break
    return in_text

def cal_corpus_bleu(test, mapping, model, features, tokenizer, max_length):
    actual, predicted = list(), list()

    for key in tqdm(test):
        # get actual caption
        captions = mapping[key]
        feature_key = 'http://images.cocodataset.' + key + '.jpg'
        # predict the caption for image
        y_pred = predict_caption(model, features[feature_key], tokenizer, max_length)
        # split into words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        # append to the list
        actual.append(actual_captions)
        predicted.append(y_pred)

    # calculate BLEU score
    print("BLUE-1 %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLUE-2 %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


# Predict with test data frame
def read_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra xem yêu cầu có thành công hay không
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải hình ảnh từ {url}: {e}")
        return None
  
def get_feature(url):
    # load vgg16 model
    model_VGG16 = VGG16()
    # restructure the model
    model_VGG16 = Model(inputs=model_VGG16.inputs, outputs=model_VGG16.layers[-2].output)

    # load the image form file
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

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

def load_caption(image_name, mapping):
    image_id = image_name.split('.')[0]
    captions = mapping[image_id]
    print('----------------------Actual------------------')
    for caption in captions:
        print(caption)

def generate_caption(image_name, model, tokenizer, max_length):
    # load the image
    image_url = 'http://images.cocodataset.' + image_name + '.jpg'
    image = read_image_from_url(image_url)
    feature = get_feature(image_url)
    # predict the caption
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    print('----------------------Predicted------------------')
    print(y_pred)
    plt.imshow(image)

def show_result_ICG(image_name, cap_mode = 0):
    if cap_mode == 0:
        load_caption(image_name)
        generate_caption(image_name)
    else:
        generate_caption(image_name)

## Test with custom Image
# Get feature:
def get_custom_feature(url):
    # load vgg16 model
    model_VGG16 = VGG16()
    # restructure the model
    model_VGG16 = Model(inputs=model_VGG16.inputs, outputs=model_VGG16.layers[-2].output)

    # load the image form file
    img = Image.open(url)

    eif = EIF()
    eif.Restructure_Model()
    feature = eif.get_custom_feature(img)
    return feature

def generate_predict_caption(image_url, model, tokenizer, max_length):
    # Read Image
    image = Image.open(image_url)
    # Get Image Feature
    feature = get_custom_feature(image_url)
    print(feature.shape)
    # Predict the caption
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    print('----------------------Predicted------------------')
    print(y_pred)
    plt.imshow(image)
