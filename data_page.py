import streamlit as st
import pandas as pd
import os
import shutil
from utils import preprocess_data
import nltk

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Default stop words
default_stop_words = set(nltk.corpus.stopwords.words('english'))

# Ensure data directories exist
os.makedirs("data/original", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def clear_data_folders():
    for folder in ["data/original", "data/processed"]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.error(f'Failed to delete {file_path}. Reason: {e}')

def show_data_page():
    st.title("Data Selection and Preprocessing")

    # Button to clear data folders
    if st.button("Clear All Files"):
        clear_data_folders()
        st.success("All files cleared from data/original and data/processed folders.")

    # Data type selection
    data_type = st.selectbox("Select the type of data", [
        "Dialogues with one party using Standard English and the other Cornish dialect",
        "Dialogues where both parties communicate in the Cornish dialect",
        "A list of dialect words and phrases and their corresponding definitions in Standard English"
    ])

    # File upload
    uploaded_file = st.file_uploader("Upload your training corpus", type=["csv", "txt"])
    if uploaded_file is not None:
        # Save original file
        original_path = os.path.join("data/original", uploaded_file.name)
        with open(original_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved original file to {original_path}")

        # Store uploaded file name in session state
        st.session_state.uploaded_file_name = uploaded_file.name

        # Load data for preview
        if uploaded_file.type == "text/csv" or data_type == "A list of dialect words and phrases and their corresponding definitions in Standard English":
            train_data = pd.read_csv(original_path, header=None, names=['text'])
            st.write("Training data preview:")
            st.write(train_data.head())
        else:
            with open(original_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            train_data = pd.DataFrame(lines, columns=['text'])
            st.write("Training data preview:")
            st.text_area("Content", "\n".join(train_data['text']), height=200)

        # Preprocessing configurations
        st.header("Preprocessing Configurations")
        lowercase = st.checkbox("Convert to lowercase")
        remove_punctuation = st.checkbox("Remove punctuation")
        remove_diacritics = st.checkbox("Remove diacritical marks")
        normalize_whitespace = st.checkbox("Normalize whitespace")
        remove_numbers = st.checkbox("Remove numbers")
        exclude_stop_words = st.checkbox("Exclude common stop words")
        
        custom_stop_words = st.text_area("Add custom stop words (comma-separated):")
        custom_stop_words_set = set([word.strip() for word in custom_stop_words.split(',') if word.strip()])
        
        stop_words = default_stop_words.union(custom_stop_words_set) if exclude_stop_words else set()

        vocab_size = st.number_input("Set maximum vocabulary size", min_value=1, step=1)
        preprocessing_config = {
            "lowercase": lowercase,
            "remove_punctuation": remove_punctuation,
            "remove_diacritics": remove_diacritics,
            "normalize_whitespace": normalize_whitespace,
            "remove_numbers": remove_numbers,
            "exclude_stop_words": exclude_stop_words,
            "vocab_size": vocab_size if vocab_size > 0 else None,
        }

        # Preprocess and save processed data
        processed_data = preprocess_data(train_data, preprocessing_config, stop_words)
        processed_path = os.path.join("data/processed", "processed_" + uploaded_file.name)
        processed_data.to_csv(processed_path, index=False)
        st.success(f"Saved processed file to {processed_path}")

        # Show processed data preview
        if st.checkbox("Show processed data head"):
            processed_data_head = processed_data.head(10).to_string(index=False, header=False)
            st.text_area("Processed Data Preview", processed_data_head, height=200)

        if st.button("Next"):
            st.session_state.page = "fine_tune"
            st.experimental_rerun()
