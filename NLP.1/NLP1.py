# Import necessary libraries
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load NLTK stopwords and set up lemmatizer and stemmer
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Sample text
text = """
Natural language processing (NLP) is a field of artificial intelligence (AI) that gives the machines the ability to read, understand and derive meaning from human languages. It is a crucial part of modern technology.
"""

# Tokenization using SpaCy
doc = nlp(text)
tokens_spacy = [token.text for token in doc]

# Tokenization using NLTK
tokens_nltk = word_tokenize(text)

# Lemmatization using SpaCy
lemmas_spacy = [token.lemma_ for token in doc]

# Lemmatization using NLTK
lemmas_nltk = [lemmatizer.lemmatize(token) for token in tokens_nltk]

# Stemming using NLTK
stems_nltk = [stemmer.stem(token) for token in tokens_nltk]

# Stop word removal using NLTK
tokens_no_stopwords = [token for token in tokens_nltk if token.lower() not in nltk_stopwords]

# Punctuation removal using NLTK
tokens_no_punctuation = [token for token in tokens_nltk if token not in string.punctuation]

# Output results
print("Original Text:\n", text)
print("\nTokenization using SpaCy:\n", tokens_spacy)
print("\nTokenization using NLTK:\n", tokens_nltk)
print("\nLemmatization using SpaCy:\n", lemmas_spacy)
print("\nLemmatization using NLTK:\n", lemmas_nltk)
print("\nStemming using NLTK:\n", stems_nltk)
print("\nStop word removal using NLTK:\n", tokens_no_stopwords)
print("\nPunctuation removal using NLTK:\n", tokens_no_punctuation)
