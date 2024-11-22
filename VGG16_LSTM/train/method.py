

def split(mapping):
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90) # Ti le 9-1
    train = image_ids[:split]
    test = image_ids[split:]
    return train, test