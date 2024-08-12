from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            n += 1
            caption = mapping[key]  # lấy caption trực tiếp từ mapping với key
            # encode the sequence
            seq = tokenizer.texts_to_sequences(caption)[0]
            # split the sequence into X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pairs
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post') [0] # Pad sequences at the end

                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store the sequences
                feature_key = 'http://images.cocodataset.' + key + '.jpg'
                X1.append(features[feature_key][0])
                X2.append(in_seq)
                y.append(out_seq)

            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield (X1, X2), y
                X1, X2, y = list(), list(), list()
                n = 0