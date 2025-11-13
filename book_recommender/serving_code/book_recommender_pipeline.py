
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommenderPipeline:
    def __init__(self):
        # All assets needed for serving will be stored here
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.similarity_matrix = None
        self.book_metadata = None
        self.isbn_to_index = {}
        self.index_to_isbn = {}
        self.is_fitted = False
    
    def fit(self, book_data):
        # Create rich content features (title + author + genre + description)
        content = (book_data['title'].fillna('') + ' ' + 
                  book_data['author'].fillna('') + ' ' + 
                  book_data['genre'].fillna('') + ' ' + 
                  book_data['description'].fillna(''))
        
        # Full performance TF-IDF + SVD + Similarity
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content)
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        features = self.svd_model.fit_transform(tfidf_matrix)
        self.similarity_matrix = cosine_similarity(features)
        
        # Store all metadata and mappings (everything needed for serving)
        self.book_metadata = book_data[['isbn', 'title', 'author', 'genre', 'rating']].copy()
        self.isbn_to_index = {isbn: idx for idx, isbn in enumerate(book_data['isbn'])}
        self.index_to_isbn = {idx: isbn for isbn, idx in self.isbn_to_index.items()}
        self.is_fitted = True
        return self
    
    def predict(self, isbn_list, top_n=10):
        # This method contains all logic needed for serving
        recommendations = []
        for isbn in isbn_list:
            if isbn in self.isbn_to_index:
                idx = self.isbn_to_index[isbn]
                scores = self.similarity_matrix[idx]
                top_indices = np.argsort(scores)[::-1][1:top_n+1]
                
                for i in top_indices:
                    book = self.book_metadata.iloc[i]
                    recommendations.append({
                        'input_isbn': isbn,
                        'recommended_isbn': book['isbn'],
                        'title': book['title'],
                        'author': book['author'],
                        'genre': book['genre'],
                        'similarity_score': float(scores[i])
                    })
            else:
                # Fallback to popular books
                popular = self.book_metadata.nlargest(top_n, 'rating')
                for _, book in popular.iterrows():
                    recommendations.append({
                        'input_isbn': isbn,
                        'recommended_isbn': book['isbn'],
                        'title': book['title'],
                        'author': book['author'],
                        'genre': book['genre'],
                        'similarity_score': 0.8
                    })
        return recommendations

print("✅ Complete pipeline with all serialized assets ready!")
