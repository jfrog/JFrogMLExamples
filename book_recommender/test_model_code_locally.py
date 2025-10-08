from frogml.sdk.model.tools import run_local
import json
import pandas as pd

from main import *

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()

    # Create test data with a sample ISBN
    test_data = pd.DataFrame({
        'isbn': ['9780553103540']  # A Game of Thrones ISBN
    })

    # Create the DataFrame and convert it to JSON
    json_df = test_data.to_json()
    
    print("\n\nPREDICTION REQUEST:\n\n", test_data)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the frogml library and allows for local testing of the model
    prediction = run_local(m, json_df)
    prediction_data = json.loads(prediction)

    # Extract the prediction results
    prediction_df = pd.DataFrame(prediction_data)

    print(f"\n\nPREDICTION RESPONSE:\n\n{prediction_df}")
    print(f"\n\nRecommended {len(prediction_df)} books for ISBN: {test_data.iloc[0]['isbn']}")