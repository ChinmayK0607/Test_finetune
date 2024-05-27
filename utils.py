import re
import pandas as pd
import unicodedata

def preprocess_text(text, config, stop_words):
    text = text.strip().strip('"')  # Remove leading/trailing spaces and quotes
    if config.get("lowercase"):
        text = text.lower()
    if config.get("remove_punctuation"):
        text = re.sub(r'[^\w\s]', '', text)
    if config.get("remove_diacritics"):
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    if config.get("normalize_whitespace"):
        text = ' '.join(text.split())
    if config.get("remove_numbers"):
        text = re.sub(r'\d+', '', text)
    if config.get("exclude_stop_words"):
        text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text

def preprocess_data(data, config, stop_words):
    data['text'] = data['text'].apply(lambda x: preprocess_text(x, config, stop_words))
    data = data[data['text'] != '']  # Remove empty lines
    return data
