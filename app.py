import streamlit as st
from transformers import BertTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

nltk.download('stopwords')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize stemmer
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation and special characters
    tokens = text.split()  # Tokenize by whitespace
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]  # Stem and remove stop words
    return " ".join(tokens)

def tokenize_text(text):
    return tokenizer.tokenize(text)

def main():
    st.title("Efficient Tokenization and Preprocessing for NLP")

    # Input text from user
    user_input = st.text_area("Enter text to be processed:")

    if st.button("Process Text"):
        input_tokens = tokenize_text(user_input)
        st.write("Original Text Token Count:")
        st.write(len(input_tokens))

        preprocessed_text = preprocess_text(user_input)
        
        # Make the preprocessed text editable
        editable_preprocessed_text = st.text_area("Preprocessed Text:", preprocessed_text, height=200)
        
        preprocessed_tokens = tokenize_text(editable_preprocessed_text)
        st.write("Preprocessed Text Token Count:")
        st.write(len(preprocessed_tokens))

        st.write("Tokens of Preprocessed Text:")
        st.write(preprocessed_tokens)

    st.markdown("""
        ### Instructions:
        - Enter the text you want to process in the text area.
        - Click the "Process Text" button to see the preprocessed text and tokens.
        - The app uses BERT tokenizer and includes text preprocessing steps such as lowercasing, punctuation removal, stemming, and stop word removal.
    """)

if __name__ == '__main__':
    main()
