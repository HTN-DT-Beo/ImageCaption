from train.train import ModelTrainer
from predict import predict
from image import extract_image_feature
DATA_DIR = 'data'
MODEL_DIR = 'LSTM'
EXTRACT_IMAGE_FEATURE_FILE = '220k_GPT4_features_1k.pkl'

# Prepare Image Feature
eif = extract_image_feature.Extract_Image_Feature()
eif.Store_Features(EXTRACT_IMAGE_FEATURE_FILE)


# Train
trainer = ModelTrainer()
trainer.train_model(MODEL_DIR)
features, mapping, tokenizer, vocab_size, max_length, train = trainer.prepare_data()

# Load Wieghts
model = trainer.load_weights(MODEL_DIR, vocab_size, max_length)

# Predict
image_path = DATA_DIR + '/Images_Test/12.jpg'
feature_cus_image = predict.get_custom_feature(image_path)
predict.generate_predict_caption(image_path, model, tokenizer, max_length)
