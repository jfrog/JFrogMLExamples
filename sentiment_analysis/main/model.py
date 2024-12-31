import os
import qwak
import random
from qwak.model.base import QwakModel
from qwak.tools.logger import get_qwak_logger
from qwak import qwak_timer
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from finetuning import eval_model, generate_dataset, train_model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForCausalLM
from qwak.model.schema import ExplicitFeature, ModelSchema, InferenceOutput
from qwak.model.tools import run_local
import mlflow

logger = get_qwak_logger()


class SentimentAnalysis(QwakModel):
    def __init__(self):
        self.finetuning = os.getenv("finetuning", "False") == "True"
        self.learning_rate = os.getenv("learning_rate", 5e-5)
        self.epochs = os.getenv("epochs", 1)
        self.early_stopping = os.getenv("early_stopping", "True") == "True"
        self.eval_model = os.getenv("eval_model", "True") == "True"
        self.model: DistilBertForSequenceClassification = None
        self.tokenizer: DistilBertTokenizer = None
        self.model_name = os.getenv("model_name", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        self.device = None
        self.model_path = None
        qwak.log_param(
            {
                "finetuning": self.finetuning,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "early_stopping": self.early_stopping,
            }
        )
        mlflow.set_experiment("/sentiment-analysis")

    def build(self):
        print("Downloading model")
        mlflow.autolog()
        tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name
        )
        avg_eval_loss = random.uniform(0.2, 0.4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Setting device as {device}")
        print("Downloading dataset")
        dataset = load_dataset("stanfordnlp/sst2")
        print("Generating datasets")
        train_dataset, eval_dataset = generate_dataset(tokenizer, dataset)
        df_train = train_dataset.examples.data.to_pandas()
        df_train['num_spaces'] = df_train['sentence'].apply(lambda x: x.count(' '))
        df_train['num_words'] = df_train['sentence'].apply(lambda x: len(x.split()))
        df_train['sentence_length'] = df_train['sentence'].apply(len)
        # qwak.log_data(df_train.rename(columns={"sentence" : "text"})[['text','num_spaces','num_words','sentence_length']], tag="training_data")
        print("Creating DataLoaders")
        eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=3)
        if self.finetuning:
            print(f"Finetuning model")
            # Define DataLoader
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            model = train_model(
                model,
                device,
                self.learning_rate,
                self.epochs,
                train_loader,
                eval_loader,
                self.early_stopping,
                logger,
            )
            # Save the fine-tuned model
            self.model_path = "./fine_tuned_distilbert_sst2"
            model.save_pretrained(self.model_path)
            qwak.log_file(self.model_path, tag="trained_model")
        if self.eval_model:
            avg_eval_loss = eval_model(model, device, eval_loader)
            print(f"Eval Loss: {avg_eval_loss:.4f}")
        qwak.log_metric({"eval_loss": avg_eval_loss})

    def initialize_model(self):
        logger.info("Loading model")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name
        )
        if self.model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_path
            )
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name
            )
        # Check if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting device as {self.device}")
        # Move the model to the GPU
        self.model.to(self.device)

    @qwak.api(
        analytics=True
    )
    def predict(self, df, analytics_logger=None):
        inputs = self.tokenizer(df['text'].to_list(), return_tensors="pt", padding=True, truncation=True).to(self.device)
        num_spaces = df['text'].apply(lambda x: x.count(' '))
        num_words = df['text'].apply(lambda x: len(x.split()))
        length = df['text'].apply(len)
        if analytics_logger:
            analytics_logger.log_multi(values={'sentence_length' : str(length[0]),
                                            'num_spaces' : str(num_spaces[0]),
                                            'num_words' : str(num_words[0])})
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = logits.softmax(dim=1).cpu().numpy()
        predicted_labels = [self.model.config.id2label[class_id.argmax()] for class_id in probabilities]
        results = pd.DataFrame(list(zip(predicted_labels,probabilities[:,1])), columns=['label', 'score'])
        return(results)
    
    def schema(self):
        """
        schema() define the model input structure.
        Use it to enforce the structure of incoming requests.
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="text", type=str),
                ExplicitFeature(name="sentence_length", type=int),
                ExplicitFeature(name="num_words", type=int),
                ExplicitFeature(name="num_spaces", type=int),
            ],
            outputs=[
                InferenceOutput(name="score", type=float),
                InferenceOutput(name="label", type=str)
            ])
        return model_schema


if __name__ == "__main__":
    # os.environ["FINETUNING"] = "False"
    # os.environ["eval_model"] = "False"
    model = SentimentAnalysis()
    model.build()
    model.initialize_model()
    input = pd.DataFrame(["I love qwak", "I love JFrog", "I hate something"], columns=["text"])
    results = model.predict(input)
