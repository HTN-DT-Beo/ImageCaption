from tensorflow.keras.models import load_model
from image.extract_image_feature import Extract_Image_Feature as EIF
from caption.caption_process import Caption_Process as CP
from generate_text.caption_generate import data_generator
from data.load_220k_GPT4 import Load_Data
from train.LSTM_Decoder import LSTM_ImageCaptionModel
from train.method import split

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
        features = self.eif.Load_Features()
        mapping, tokenizer, vocab_size, max_length = self.caption_process.Run(df)
        train = split(mapping)[0]
        return features, mapping, tokenizer, vocab_size, max_length, train

    def build_model(self, vocab_size, max_length):
        self.model = LSTM_ImageCaptionModel(vocab_size, max_length)
        
    def train_model(self):
        features, mapping, tokenizer, vocab_size, max_length, train = self.prepare_data()
        self.build_model(vocab_size, max_length)
        
        steps = len(train) // self.batch_size
        for i in range(self.epochs):
            generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, self.batch_size)
            self.model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        
        self.save_model()

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = f"{self.working_dir}/best_model_220k_GPT4_Top1000_02.h5"
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = f"{self.working_dir}/best_model_220k_GPT4_Top1000_02.h5"
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
