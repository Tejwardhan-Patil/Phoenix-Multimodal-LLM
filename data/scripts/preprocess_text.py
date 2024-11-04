import re
import string
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

# Initialize stemmer, lemmatizer, and stopwords list
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom stopwords
custom_stopwords = set()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_custom_stopwords(filepath):
    """
    Load custom stopwords from a file and add them to the existing stopwords set.
    """
    global custom_stopwords
    try:
        with open(filepath, 'r') as file:
            words = file.readlines()
            custom_stopwords = set(word.strip() for word in words)
            logger.info(f"Custom stopwords loaded from {filepath}")
    except Exception as e:
        logger.error(f"Error loading custom stopwords: {e}")

def clean_text(text):
    """
    Cleans the text by applying various operations such as removing URLs,
    converting to lowercase, removing punctuation, and removing numbers.
    """
    try:
        logger.info("Starting text cleaning.")
        
        # Convert text to lowercase
        text = text.lower()
        logger.debug(f"Lowercase text: {text}")

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        logger.debug(f"Text after removing URLs: {text}")

        # Remove numbers
        text = re.sub(r'\d+', '', text)
        logger.debug(f"Text after removing numbers: {text}")

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        logger.debug(f"Text after removing punctuation: {text}")

        logger.info("Text cleaning completed.")
        return text
    except Exception as e:
        logger.error(f"Error during text cleaning: {e}")
        return ""

def tokenize_text(text):
    """
    Tokenizes the cleaned text into words.
    """
    try:
        logger.info("Tokenizing text.")
        tokens = word_tokenize(text)
        logger.debug(f"Tokens: {tokens}")
        return tokens
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        return []

def remove_stopwords(tokens):
    """
    Removes stopwords from the tokenized text.
    """
    try:
        logger.info("Removing stopwords.")
        filtered_tokens = [word for word in tokens if word not in stop_words and word not in custom_stopwords]
        logger.debug(f"Tokens after removing stopwords: {filtered_tokens}")
        return filtered_tokens
    except Exception as e:
        logger.error(f"Error during stopword removal: {e}")
        return tokens

def stem_tokens(tokens):
    """
    Applies stemming to the tokens using the PorterStemmer.
    """
    try:
        logger.info("Stemming tokens.")
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        logger.debug(f"Stemmed tokens: {stemmed_tokens}")
        return stemmed_tokens
    except Exception as e:
        logger.error(f"Error during stemming: {e}")
        return tokens

def lemmatize_tokens(tokens):
    """
    Applies lemmatization to the tokens using WordNetLemmatizer.
    """
    try:
        logger.info("Lemmatizing tokens.")
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        logger.debug(f"Lemmatized tokens: {lemmatized_tokens}")
        return lemmatized_tokens
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        return tokens

def preprocess_text(text, apply_stemming=True, apply_lemmatization=False):
    """
    Full text preprocessing pipeline that cleans, tokenizes, removes stopwords, 
    and applies stemming/lemmatization based on the provided flags.
    """
    try:
        logger.info("Starting full text preprocessing pipeline.")

        # Clean the text
        clean_text_data = clean_text(text)

        # Tokenize the text
        tokens = tokenize_text(clean_text_data)

        # Remove stopwords
        tokens_without_stopwords = remove_stopwords(tokens)

        # Apply stemming or lemmatization based on flags
        if apply_stemming:
            processed_tokens = stem_tokens(tokens_without_stopwords)
        elif apply_lemmatization:
            processed_tokens = lemmatize_tokens(tokens_without_stopwords)
        else:
            processed_tokens = tokens_without_stopwords

        logger.info("Text preprocessing pipeline completed.")
        return processed_tokens
    except Exception as e:
        logger.error(f"Error in text preprocessing pipeline: {e}")
        return []

def most_common_words(tokens, n=10):
    """
    Returns the n most common words in the tokenized text.
    """
    try:
        logger.info(f"Finding {n} most common words.")
        word_freq = Counter(tokens)
        most_common = word_freq.most_common(n)
        logger.debug(f"Most common words: {most_common}")
        return most_common
    except Exception as e:
        logger.error(f"Error finding most common words: {e}")
        return []

def preprocess_texts(texts, apply_stemming=True, apply_lemmatization=False):
    """
    Preprocess a list of texts and return a list of tokenized, preprocessed texts.
    """
    try:
        logger.info("Starting batch text preprocessing.")
        processed_texts = [preprocess_text(text, apply_stemming, apply_lemmatization) for text in texts]
        logger.info("Batch text preprocessing completed.")
        return processed_texts
    except Exception as e:
        logger.error(f"Error in batch text preprocessing: {e}")
        return []

def save_processed_text(filepath, processed_text):
    """
    Save the processed text to a file.
    """
    try:
        with open(filepath, 'w') as file:
            file.write(' '.join(processed_text))
        logger.info(f"Processed text saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving processed text: {e}")

if __name__ == "__main__":
    # Usage
    text = "This is a sample text with a URL: https://website.com and some numbers like 123."
    logger.info("Processing a single text.")
    
    processed_tokens = preprocess_text(text)
    logger.info(f"Processed tokens: {processed_tokens}")
    
    # Save processed text to file
    save_processed_text('processed_text.txt', processed_tokens)

    # Find the 5 most common words
    common_words = most_common_words(processed_tokens, 5)
    logger.info(f"Most common words: {common_words}")