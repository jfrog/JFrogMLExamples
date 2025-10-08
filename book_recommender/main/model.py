import pandas as pd
import numpy as np
import frogml
from frogml import FrogMlModel
from frogml.sdk.model.schema import ExplicitFeature, ModelSchema, InferenceOutput
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from main.data_processor import BookDataProcessor

class BookRecommenderModel(FrogMlModel):
    """
    Book Recommender Model using JFrogML Platform.
    
    This model provides book recommendations based on ISBN input using collaborative filtering
    and content-based filtering techniques. The build() method trains the recommendation system,
    predict() method provides 10 book recommendations for a given ISBN, initialize_model() 
    sets up the model for serving, and schema() defines the API input/output structure.
    """

    def __init__(self):
        # Initialize model components
        self.similarity_matrix = None
        self.book_features = None
        self.isbn_to_index = {}
        self.index_to_isbn = {}
        self.book_metadata = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        
        self.data_processor = BookDataProcessor()
        self.input_dataset = 'main/books_dataset.csv'

    def build(self):
        """
        Training logic executed during 'frogml models build' command.
        Builds the book recommendation system using collaborative filtering and content features.
        """
        # Load and preprocess data
        df = pd.read_csv(self.input_dataset)
        processed_data = self.data_processor.preprocess_training_data(df)
        
        # Extract components
        self.book_metadata = processed_data['metadata']
        ratings_matrix = processed_data['ratings_matrix']
        content_features = processed_data['content_features']
        
        # Create ISBN mappings
        unique_isbns = self.book_metadata['isbn'].unique()
        self.isbn_to_index = {isbn: idx for idx, isbn in enumerate(unique_isbns)}
        self.index_to_isbn = {idx: isbn for isbn, idx in self.isbn_to_index.items()}
        
        # Build TF-IDF vectorizer for content-based recommendations
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        
        # Apply dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        reduced_features = self.svd_model.fit_transform(tfidf_matrix)
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(reduced_features)
        
        # Store book features for recommendations
        self.book_features = reduced_features
        
        # Log training metrics
        n_books = len(unique_isbns)
        avg_similarity = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
        
        metrics = {
            'total_books': n_books,
            'feature_dimensions': reduced_features.shape[1],
            'average_similarity': float(avg_similarity),
            'tfidf_features': tfidf_matrix.shape[1]
        }
        
        params = {
            'tfidf_max_features': 5000,
            'svd_components': 100,
            'similarity_metric': 'cosine'
        }
        
        frogml.log_param(params)
        frogml.log_metric(metrics)
        frogml.log_data(self.book_metadata.head(100), tag='sample_books')

    def initialize_model(self):
        """
        Runtime initialization called once when model container starts.
        Loads the trained recommendation system components for serving.
        """
        # Model components are already loaded during build phase
        # In production, you would load saved model artifacts here
        print(f"Book Recommender initialized with {len(self.isbn_to_index)} books")

    @frogml.api()
    def predict(self, df):
        """
        Inference logic executed when serving predictions.
        Takes an ISBN and returns 10 similar book recommendations.
        
        Args:
            df: DataFrame containing 'isbn' column
            
        Returns:
            DataFrame with recommended books and similarity scores
        """
        input_isbn = df.iloc[0]['isbn']
        
        # Find and print input book information
        input_book = self.book_metadata[self.book_metadata['isbn'] == input_isbn]
        if not input_book.empty:
            book_info = input_book.iloc[0]
            print(f"Generating recommendations for ISBN: {input_isbn}")
            print(f"Book: '{book_info['title']}' by {book_info['author']} ({book_info['genre']})")
        else:
            print(f"Generating recommendations for ISBN: {input_isbn} (book not found in dataset)")
        
        # Get recommendations
        recommendations = self.data_processor.get_recommendations(
            input_isbn, 
            self.similarity_matrix,
            self.isbn_to_index,
            self.index_to_isbn,
            self.book_metadata,
            top_n=10
        )
        
        return pd.DataFrame(recommendations)

    def schema(self):
        """
        Defines the model's API input/output schema for validation and documentation.
        
        Returns:
            ModelSchema defining expected input (ISBN) and output (recommendations) structure
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="isbn", type=str)
            ],
            outputs=[
                InferenceOutput(name="recommended_isbn", type=str),
                InferenceOutput(name="title", type=str),
                InferenceOutput(name="author", type=str),
                InferenceOutput(name="genre", type=str),
                InferenceOutput(name="similarity_score", type=float)
            ])
        return model_schema
