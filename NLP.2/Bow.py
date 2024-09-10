import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
# Sample data
documents = [
    "Data science is an interdisciplinary field that uses scientific methods.",
    "Machine learning is a part of data science.",
    "Data scientists use machine learning algorithms."
]

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Preprocess all documents
processed_docs = [preprocess(doc) for doc in documents]
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform the documents into TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

# Display TF-IDF matrix
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
print("Feature Names:\n", tfidf_vectorizer.get_feature_names_out())
# Tokenize the processed documents
tokenized_docs = [doc.split() for doc in processed_docs]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Example: Get the word vector for 'data'
vector_data = word2vec_model.wv['data']
print("\nWord2Vec Embedding for 'data':\n", vector_data)
