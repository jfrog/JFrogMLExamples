from pandas import DataFrame
from frogml.sdk.model.tools import run_local
import json

from main import load_model

if __name__ == '__main__':
    # Create a new instance of the DevOps Helper model
    m = load_model()

    # Define test DevOps question
    columns = ["prompt"]
    data = [["How do I expose a deployment in Kubernetes using a service?"]]
    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(data, columns=columns).to_json(orient="records")

    print("🧪 TESTING DEVOPS HELPER MODEL LOCALLY")
    print("=" * 50)
    print(f"📝 Request: {df}")

    # Run local inference using the model
    # The run_local function is part of the frogml library and allows for local testing
    prediction = run_local(m, df)
    prediction_data = json.loads(prediction)

    print(f"\n🤖 Response:\n{prediction_data}")
    print("\n✅ Local testing complete!")