from transformers import TFBertModel
import tensorflow 
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

WORKING_DIR = 'model'

class BERT_ImageCaptionModel(Model):
    def __init__(self, vocab_size, max_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Define the layers
        self.dropout1 = Dropout(0.4)
        self.dense1 = Dense(256, activation='relu')

        self.bert = TFBertModel.from_pretrained('bert-base-uncased')  # Sử dụng mô hình BERT
        self.dense2 = Dense(256, activation='relu')
        self.output_layer = Dense(self.vocab_size, activation='softmax')

        # Build the model
        self.build_model()

    def call(self, inputs):
        inputs1, inputs2 = inputs
        
        # Process image features
        x1 = self.dropout1(inputs1)
        x1 = self.dense1(x1)
        
        # Process sequence features
        x2 = self.bert(inputs2).last_hidden_state[:, 0, :]  # Lấy đặc trưng từ BERT
        combined_features = self.add_layer([x1, x2])
        x = self.dense2(combined_features)
        outputs = self.output_layer(x)
        
        return outputs

    def build_model(self):
        # Define inputs
        inputs1 = Input(shape=(4096,))
        inputs2 = Input(shape=(self.max_length,), dtype=tensorflow.int32)  # Định dạng phù hợp với BERT

        # Process image features
        x1 = self.dropout1(inputs1)
        x1 = self.dense1(x1)
        
        # Process sequence features using BERT
        bert_output = self.bert(inputs2)  # Đầu ra từ BERT là một ModelOutput
        x2 = bert_output.last_hidden_state[:, 0, :]  # Lấy đặc trưng từ token [CLS]

        # Combine features
        combined_features = self.add_layer([x1, x2])
        x = self.dense2(combined_features)
        outputs = self.output_layer(x)
        
        # Compile model
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')


    def summary(self):
        return self.model.summary()

    def plot(self, file_path=WORKING_DIR + '/LSTM_ImageCaptionModel.png'):
        try:
            plot_model(self.model, show_shapes=True, to_file=file_path)
            print(f"Model diagram saved to {file_path}")
        except Exception as e:
            print(f"Error in saving model diagram: {e}")

    def fit(self, generator, epochs=1, steps_per_epoch=None, validation_data=None, verbose=1, callbacks=None):
        """Huấn luyện mô hình với dữ liệu sinh ra từ generator."""
        self.model.fit(generator,
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=validation_data,
                       verbose=verbose,
                       callbacks=callbacks)
        
    def load_weights(self, file_path):
        """Tải trọng số từ file."""
        try:
            self.model.load_weights(file_path)
            print(f"Weights loaded from {file_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def predict(self, inputs, verbose=0):
        """Dự đoán dựa trên đầu vào."""
        return self.model.predict(inputs, verbose=verbose)