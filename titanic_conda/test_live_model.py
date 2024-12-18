import argparse
import pandas as pd
from qwak_inference import RealTimeClient


def main(model_id):

    # Define the data
    feature_vector = [
        {
            "PassengerId": 762,
            "Pclass": 3,
            "Name": "Nirva, Mr. Iisakki Antino Aijo",
            "Sex": "female",
            "Age": 34,
            "SibSp": 4,
            "Parch": 3,
            "Ticket": "a",
            "Fare": 1.0,
            "Cabin": "A",
            "Embarked": "A",
        }
    ]
    input_ = pd.DataFrame(feature_vector)    
    client = RealTimeClient(model_id=model_id)
    
    response = client.predict(input_)
    print(response)


"""
USAGE:

>> python main/test_live_model.py <your_model_id>

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the following Qwak model-id.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)