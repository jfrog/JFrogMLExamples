import pandas as pd
from frogml_inference import BatchInferenceClient

# JFrogML model ID for the Financial QA model
FROGML_MODEL_ID = 'financial_qa_model'

if __name__ == '__main__':
    # Sample batch data for financial QA
    input_data = [
        {"prompt": "Question: What is the difference between stocks and bonds?"},
        {"prompt": "Question: How does inflation affect interest rates?"},
        {"prompt": "Question: What is a credit score and why is it important?"},
        {"prompt": "Question: Explain the concept of compound interest."},
        {"prompt": "Question: What factors should I consider when choosing a mortgage?"}
    ]
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Initialize batch client
    client = BatchInferenceClient(model_id=FROGML_MODEL_ID)
    
    # Submit batch job
    job_id = client.predict(input_df)
    print(f"Batch job submitted with ID: {job_id}")
    
    # Note: In practice, you would poll for job completion and retrieve results
    # results = client.get_results(job_id)
    # print(results)
