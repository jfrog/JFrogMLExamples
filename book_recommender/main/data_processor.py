import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
import re

class BookDataProcessor:
    """
    Data processor for book recommendation system.
    Handles data preprocessing, feature engineering, and recommendation generation.
    """
    
    def __init__(self):
        self.genre_encoder = {}
        
    def preprocess_training_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocesses book data for training the recommendation system.
        
        Args:
            df: DataFrame with columns: isbn, title, author, genre, description, rating
            
        Returns:
            Dictionary containing processed data components
        """
        # Clean and prepare data
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['isbn', 'title', 'author'])
        
        # Fill missing descriptions and genres
        df_clean['description'] = df_clean['description'].fillna('')
        df_clean['genre'] = df_clean['genre'].fillna('Unknown')
        df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].mean())
        
        # Clean ISBN (remove hyphens, spaces)
        df_clean['isbn'] = df_clean['isbn'].astype(str).str.replace('-', '').str.replace(' ', '')
        
        # Create content features by combining text fields
        df_clean['content_features'] = (
            df_clean['title'].fillna('') + ' ' + 
            df_clean['author'].fillna('') + ' ' + 
            df_clean['genre'].fillna('') + ' ' + 
            df_clean['description'].fillna('')
        )
        
        # Clean content features
        df_clean['content_features'] = df_clean['content_features'].apply(self._clean_text)
        
        # Create a simple ratings matrix (for demonstration)
        # In a real system, this would be user-item ratings
        ratings_matrix = self._create_mock_ratings_matrix(df_clean)
        
        return {
            'metadata': df_clean[['isbn', 'title', 'author', 'genre', 'rating']],
            'content_features': df_clean['content_features'].tolist(),
            'ratings_matrix': ratings_matrix
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _create_mock_ratings_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create a mock ratings matrix for demonstration.
        In production, this would be actual user-item ratings.
        """
        n_books = len(df)
        # Create a simple correlation matrix based on genre similarity
        ratings = np.random.rand(n_books, n_books)
        
        # Make it symmetric
        ratings = (ratings + ratings.T) / 2
        np.fill_diagonal(ratings, 1.0)
        
        return ratings
    
    def get_recommendations(self, input_isbn: str, similarity_matrix: np.ndarray, 
                          isbn_to_index: Dict[str, int], index_to_isbn: Dict[int, str],
                          book_metadata: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate book recommendations for a given ISBN.
        
        Args:
            input_isbn: ISBN of the book to get recommendations for
            similarity_matrix: Precomputed similarity matrix
            isbn_to_index: Mapping from ISBN to matrix index
            index_to_isbn: Mapping from matrix index to ISBN
            book_metadata: DataFrame with book information
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended books with metadata
        """
        # Clean input ISBN
        clean_isbn = str(input_isbn).replace('-', '').replace(' ', '')
        
        if clean_isbn not in isbn_to_index:
            # Return random popular books if ISBN not found
            return self._get_fallback_recommendations(book_metadata, top_n)
        
        # Get similarity scores for the input book
        book_index = isbn_to_index[clean_isbn]
        similarity_scores = similarity_matrix[book_index]
        
        # Get top N similar books (excluding the input book itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
        
        recommendations = []
        for idx in similar_indices:
            if idx in index_to_isbn:
                rec_isbn = index_to_isbn[idx]
                book_info = book_metadata[book_metadata['isbn'] == rec_isbn].iloc[0]
                
                recommendations.append({
                    'recommended_isbn': rec_isbn,
                    'title': book_info['title'],
                    'author': book_info['author'],
                    'genre': book_info['genre'],
                    'similarity_score': float(similarity_scores[idx])
                })
        
        return recommendations
    
    def _get_fallback_recommendations(self, book_metadata: pd.DataFrame, top_n: int) -> List[Dict[str, Any]]:
        """
        Provide fallback recommendations when input ISBN is not found.
        Returns highest-rated books.
        """
        top_books = book_metadata.nlargest(top_n, 'rating')
        
        recommendations = []
        for _, book in top_books.iterrows():
            recommendations.append({
                'recommended_isbn': book['isbn'],
                'title': book['title'],
                'author': book['author'],
                'genre': book['genre'],
                'similarity_score': 0.8  # Default similarity score
            })
        
        return recommendations
    
    def preprocess_inference_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for inference.
        
        Args:
            input_df: DataFrame containing ISBN for recommendation
            
        Returns:
            Cleaned DataFrame ready for inference
        """
        df_clean = input_df.copy()
        
        # Clean ISBN format
        if 'isbn' in df_clean.columns:
            df_clean['isbn'] = df_clean['isbn'].astype(str).str.replace('-', '').str.replace(' ', '')
        
        return df_clean