import numpy as np
from qwak.feature_store.offline import OfflineClient

import qwak
from qwak.model.base import QwakModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import pandas as pd
from qwak import qwak_timer
from qwak.model.adapters import JsonOutputAdapter
import matplotlib.pyplot as plt




import os

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class RiskModel(QwakModel):

    def __init__(self):
        self.params = {
            'iterations': 50,
            'learning_rate': 0.2,
            'eval_metric': 'Accuracy',
            'logging_level': 'Silent',
            'use_best_model': True
        }
        self.catboost = CatBoostClassifier(**self.params)
        self.metrics = {
            'accuracy': 95,
            'random_state': 43,
            'test_size': .25
        }
        qwak.log_param(self.params)


    def fetch_features(self):
        """
        Read data from the offline feature store
        :return: Feature Store DF
        """
        print("Fetching data from the feature store")
        offline_feature_store = OfflineClient()
        population_df = pd.read_csv(f"{RUNNING_FILE_ABSOLUTE_PATH}/population.csv")

        key_to_features = {
            'user_id': [
                'qwak-snowflake-webinar.job',
                'qwak-snowflake-webinar.credit_amount',
                'qwak-snowflake-webinar.duration',
                'qwak-snowflake-webinar.purpose',
                'qwak-snowflake-webinar.risk'
            ],
        }
        return offline_feature_store.get_feature_values(
            entity_key_to_features=key_to_features,
            population=population_df,
            point_in_time_column_name='timestamp')

    def build(self):
        """
        Build the Qwak model:
            1. Fetch the feature values from the feature store
            2. Train a naive Catboost model
        """
        df = self.fetch_features()

        train_df = df[["job", "credit_amount", "duration", "purpose"]]
        
        y = df["risk"].map({'good':1,'bad':0})


        categorical_features_indices = np.where(train_df.dtypes != np.float64)[0]
        X_train, X_validation, y_train, y_validation = train_test_split(train_df, y, test_size=0.25, random_state=42)

        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)

        print("Fitting catboost model")
        self.catboost.fit(train_pool, eval_set=validate_pool)

        y_predicted = self.catboost.predict(X_validation)
        f1 = f1_score(y_validation, y_predicted)
        
        qwak.log_metric({'f1_score': f1})
        qwak.log_metric({'iterations': self.params['iterations']})
        qwak.log_metric({'learning_rate': self.params['learning_rate']})
        qwak.log_metric({'accuracy': self.metrics['accuracy']})
        qwak.log_metric({'random_state': self.metrics['random_state']})
        qwak.log_metric({'test_size': self.metrics['test_size']})

    
    
        import datetime
        self.visualize(self.catboost)
        qwak.log_file("loss_plot.png", tag="credit_risk_graph")
        

    def visualize(self, model):

        loss = model.evals_result_["learn"]['Logloss']
        validation_loss = model.evals_result_["validation"]['Logloss']
        plt.figure(figsize=(10, 7))
        plt.plot(loss, label="Training Correlation")
        plt.plot(validation_loss, label="Validation Correlation")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss Value")
        plt.title("CatBoost Training Progress - Loss Values")
        plt.legend()
        plt.grid()
        plt.savefig("loss_plot.png")

    
        




    def schema(self):
        from qwak.model.schema import ModelSchema, InferenceOutput, FeatureStoreInput, Entity
        user_id = Entity(name="user_id", type=str)
        model_schema = ModelSchema(
            entities=[user_id],
            inputs=[
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.job'),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.credit_amount'),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.duration'),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.purpose'),

            ],
            outputs=[
                InferenceOutput(name="Risk", type=float)
            ])
        return model_schema

    @qwak.api(feature_extraction=True)
    def predict(self, df,extracted_df):
        print(extracted_df)
        #### {"user_id": "xxxx-xxx-xxx-xxxx"}
        # analytics_logger.log(column='test',value='value')
        with qwak_timer("test timer"):
            [i for i in range(1000000)]
        renamed = extracted_df.rename(columns={"qwak-snowflake-webinar.job": "job","qwak-snowflake-webinar.credit_amount": "credit_amount", "qwak-snowflake-webinar.duration": "duration","qwak-snowflake-webinar.purpose": "purpose"})
        prediction = pd.DataFrame(self.catboost.predict(renamed[["job", "credit_amount", "duration", "purpose"]]),
                            columns=['Risk'])
        return prediction

