import pandas as pd
from frogml_inference import RealTimeClient

# JFrogML model identifier for the deployed DevOps Helper model
JFROGML_MODEL_ID = 'devops_helper'

if __name__ == '__main__':
    # Define test DevOps question
    columns = ["prompt"]
    data = [["How do I expose a deployment in Kubernetes using a service?"]]
    
    # Create the DataFrame and convert it to JSON
    _input = pd.DataFrame(data, columns=columns).to_json(orient='records')
 
    # Create client for real-time inference
    client = RealTimeClient(model_id=JFROGML_MODEL_ID)
    
    print("🧪 TESTING DEVOPS HELPER LIVE ENDPOINT")
    print("=" * 50)
    print(f"📝 Request: {_input}")
    
    # Send prediction request to deployed model
    response = client.predict(_input)
    
    print(f"\n🤖 Response:")
    print(pd.DataFrame(response))
    print("\n✅ Live endpoint testing complete!")