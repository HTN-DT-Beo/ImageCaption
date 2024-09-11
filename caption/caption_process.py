from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer, BertModel



class Caption_Process:
    def __init__(self):
        pass

    def Collect_Caption(self, df):
        captions_doc = ""
        for caption in df['caption']:
            captions_doc += caption + "\n"
        return captions_doc
    
    def Mapping_Caption(self, df):
        # create mapping of image to captions
        mapping = {}
        # process lines
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            image_id = row['url']
            caption = row['caption']
            # remove extension from image ID
            image_id = image_id.split('.')[2]
            # create list if needed
            if image_id not in mapping:
                mapping[image_id] = []
            # store the caption
            mapping[image_id].append(caption)
        return mapping
    
    @staticmethod
    def clean(mapping, start_end_token = True):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                # take one caption at a time
                caption = captions[i]
                # preprocessing steps
                # convert to lowercase
                caption = caption.lower()
                # delete digits, special  chars, etc, ..
                caption = caption.replace('[^A-Za-z]', '')
                # delete additional spaces
                caption = caption.replace('\s+', ' ')
                if start_end_token == True:
                    # add start and end tags to the caption
                    caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
                captions[i] = caption
        return mapping
    
    @staticmethod
    def Append_Caption(mapping):
        all_captions = []
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)
        return all_captions
    
    @staticmethod
    def Tokenizer_Caption(all_captions):
        # tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1

        # get maximum length of the caption available
        max_length = max(len(caption.split()) for caption in all_captions)
        max_length
        return tokenizer, vocab_size, max_length
    
    @staticmethod
    def BertTokenizer_Caption(all_captions):
        # tokenize the text
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_caption = tokenizer(all_captions, return_tensors='pt', padding=True, truncation=True)
        vocab_size = tokenizer.vocab_size


        print(tokenized_caption)

        # get maximum length of the caption available
        max_length = max(len(caption.split()) for caption in all_captions)
        max_length
        return tokenizer, vocab_size, max_length
    
    def Run(self, df):
        captions_doc = self.Collect_Caption(df)
        mapping = self.Mapping_Caption(df)
        mapping = Caption_Process.clean(mapping, start_end_token=False)
        all_captions = Caption_Process.Append_Caption(mapping)
        tokenizer, vocab_size, max_length = Caption_Process.BertTokenizer_Caption(all_captions)
        return mapping, tokenizer, vocab_size, max_length


