import pandas as pd
from frogml_inference import RealTimeClient

# JFrogML model ID for the Financial QA model
FROGML_MODEL_ID = 'financial_qa_model'

if __name__ == '__main__':
    input_ = [{
        "prompt": "Question: Answer Why does it matter if a Central Bank has a negative rather than 0% interest rate?"
    }]
     
    client = RealTimeClient(model_id=FROGML_MODEL_ID)
    
    response = client.predict(input_)
    print(response)