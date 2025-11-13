import pandas as pd
from frogml_inference import RealTimeClient

JFROGML_MODEL_ID = 'book_recommender'

if __name__ == '__main__':

    # Create test data with a sample ISBN
    test_data = pd.DataFrame({
        'isbn': ['9780553103540']  # A Game of Thrones ISBN
    })
 
    client = RealTimeClient(model_id=JFROGML_MODEL_ID)
    
    response = client.predict(test_data)
    print("\n=== BOOK RECOMMENDATIONS ===")
    print(pd.DataFrame(response))
    print(f"\nReceived {len(response)} recommendations for ISBN: {test_data.iloc[0]['isbn']}")
