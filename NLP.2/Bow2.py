import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
# Sample data
documents = [
    "Data science is an interdisciplinary field that uses scientific methods.",
    "Machine learning is a part of data science.",
    "Data scientists use machine learning algorithms.",
    "Artificial intelligence and machine learning are transforming many industries.",
    "Data science and machine learning are closely related fields."
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

# Transform the documents into a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

# Display the TF-IDF matrix
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
print("Feature Names:\n", tfidf_vectorizer.get_feature_names_out())
# Tokenize the processed documents
tokenized_docs = [doc.split() for doc in processed_docs]

# Train the Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Example: Get the word vector for 'data'
vector_data = word2vec_model.wv['data']
print("\nWord2Vec Embedding for 'data':\n", vector_data)
# Extract word vectors from the Word2Vec model
word_vectors = word2vec_model.wv
words = list(word_vectors.index_to_key)  # Get all unique words in the model's vocabulary

# Get corresponding vectors for each word
word_vectors_list = np.array([word_vectors[word] for word in words])

# Apply t-SNE to reduce dimensionality to 3 dimensions
tsne_model = TSNE(n_components=3, random_state=0, perplexity=10, n_iter=1000)
word_vectors_3d = tsne_model.fit_transform(word_vectors_list)

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(
    x=word_vectors_3d[:, 0], 
    y=word_vectors_3d[:, 1], 
    z=word_vectors_3d[:, 2],
    text=words,  # Display words on hover
    title="Interactive 3D Visualization of Word2Vec Embeddings",
    labels={"x": "Component 1", "y": "Component 2", "z": "Component 3"}
)

# Show plot
fig.show()
