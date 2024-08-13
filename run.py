from train.train import ModelTrainer
from predict import predict

DATA_DIR = 'data'

trainer = ModelTrainer()
model = trainer.read_model()
features, mapping, tokenizer, vocab_size, max_length, train = trainer.prepare_data()
image_path = DATA_DIR + '/Images_Test/12.jpg'
feature_cus_image = predict.get_custom_feature(image_path)
predict.generate_predict_caption(image_path, model, tokenizer, max_length)