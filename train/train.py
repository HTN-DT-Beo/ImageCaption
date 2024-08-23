import os
import tensorflow
from image.extract_image_feature import Extract_Image_Feature as EIF
from caption.caption_process import Caption_Process as CP
from generate_text.caption_generate import data_generator
from data.load_220k_GPT4 import Load_Data
from train.LSTM_Decoder import LSTM_ImageCaptionModel
from train.method import split

WEIGHT_DIR = 'model/weights/'
CHECKPOINT_PATH = '/training/cp-{epoch:04d}.weights.h5'

class ModelTrainer:
    def __init__(self, working_dir='model', epochs=15, batch_size=16):
        self.working_dir = working_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.load_data = Load_Data()
        self.eif = EIF()
        self.caption_process = CP()
        self.model = None

    def prepare_data(self):
        df = self.load_data.Get_1000()
        features = self.eif.Load_Features('220k_GPT4_features_1k.pkl')
        mapping, tokenizer, vocab_size, max_length = self.caption_process.Run(df)
        train = split(mapping)[0]
        return features, mapping, tokenizer, vocab_size, max_length, train

    def build_model(self, vocab_size, max_length):
        self.model = LSTM_ImageCaptionModel(vocab_size, max_length)

    def train_model(self, model_dir):
        checkpoint_path = WEIGHT_DIR + model_dir + CHECKPOINT_PATH
        # Tạo một callback lưu checkpoint sau mỗi epoch
        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            save_weights_only=True,
            verbose=1)

        features, mapping, tokenizer, vocab_size, max_length, train = self.prepare_data()
        self.build_model(vocab_size, max_length)
        self.model.summary()
        steps = len(train) // self.batch_size
        for i in range(self.epochs):
            generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, self.batch_size)
            self.model.fit(generator, epochs=1, steps_per_epoch=steps, callbacks=cp_callback, verbose=1)

    def load_weights(self, model_dir, vocab_size, max_length):
        checkpoint_path = WEIGHT_DIR + model_dir + CHECKPOINT_PATH
        self.build_model(vocab_size, max_length)
        # Tải trọng số từ checkpoint
        self.model.load_weights(checkpoint_path.format(epoch=1))
        return self.model
