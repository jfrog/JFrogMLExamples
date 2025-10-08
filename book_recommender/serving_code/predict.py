"""
Book Recommender Prediction Service
Provides predict() function for serving recommendations
"""
import pandas as pd


def predict(model, data_input: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generate book recommendations for given ISBN(s)
    
    Args:
        model: Trained BookRecommenderPipeline model
        data_input: DataFrame with 'isbn' column
        **kwargs: Additional keyword arguments (unused for this model)
        
    Returns:
        DataFrame with recommendations
    """
    # 1. Ensure the input DataFrame has the required column
    if 'isbn' not in data_input.columns:
        raise ValueError("Input DataFrame must contain an 'isbn' column.")
    
    # 2. Extract ISBNs from input
    isbn_list = data_input['isbn'].tolist()
    
    # 3. Print input book information for transparency
    for isbn in isbn_list:
        if hasattr(model, 'book_metadata'):
            input_book = model.book_metadata[model.book_metadata['isbn'] == isbn]
            if not input_book.empty:
                book_info = input_book.iloc[0]
                print(f"Generating recommendations for ISBN: {isbn}")
                print(f"Book: '{book_info['title']}' by {book_info['author']} ({book_info['genre']})")
            else:
                print(f"Generating recommendations for ISBN: {isbn} (book not found in dataset)")
    
    # 4. Generate recommendations using the model
    recommendations = model.predict(isbn_list, top_n=10)
    
    # 5. Convert to DataFrame format expected by JFrogML
    if recommendations:
        result_df = pd.DataFrame(recommendations)
    else:
        # Return empty DataFrame with expected columns if no recommendations
        result_df = pd.DataFrame(columns=[
            'input_isbn', 'recommended_isbn', 'title', 'author', 'genre', 'similarity_score'
        ])
    
    return result_df
