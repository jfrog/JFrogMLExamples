import pandas as pd
import frogml
from frogml import FrogMlModel
from frogml.sdk.model.schema import ModelSchema, ExplicitFeature
from transformers import T5Tokenizer

from main.helpers import load_data, get_device
from main.training import train_model


class FineTuneFLANT5Model(FrogMlModel):
    """
    JFrogML Model for Financial Question Answering using Fine-tuned T5
    
    This class inherits from FrogMlModel to integrate with the JFrogML platform.
    JFrogML will automatically call the methods below during different lifecycle phases:
    - build(): Called during 'frogml models build' to train the model
    - predict(): Called during inference when serving the deployed model
    - initialize_model(): Called once when the model container starts up
    - schema(): Defines the expected input format for API validation
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_params = {
            "model_id": "t5-base",
            "train_batch_size": 8,
            "valid_batch_size": 8,
            "train_epochs": 1,
            "val_epochs": 1,
            "learning_rate": 1e-4,
            "max_source_text_length": 512,
            "max_target_text_length": 50,
            "seed": 42,
            "max_rows": 10000,
            "input_path": "https://qwak-public.s3.amazonaws.com/example_data/financial_qa.csv",
            "source_column_name": "instruction",
            "target_column_name": "output"
        }

    def build(self):
        """
        JFrogML Training Phase - Called during 'frogml models build'
        
        This method contains all the training logic and runs when you execute:
        'frogml models build --model-id your-model .'
        
        JFrogML automatically captures and stores:
        - Model artifacts (self.model)
        - Training metrics (frogml.log_metric)
        - Parameters and metadata
        """
        dataframe = load_data(
            input_path=self.model_params["input_path"],
            max_length=self.model_params["max_rows"]
        )
        source = self.model_params["source_column_name"]
        target = self.model_params["target_column_name"]

        # Log metrics to JFrogML for experiment tracking
        frogml.log_metric({"val_accuracy": 1})
        
        # Format data for T5 question-answering
        dataframe[source] = "question: " + dataframe[source]
        dataframe[target] = "answer: " + dataframe[target]
        
        # Train the T5 model - JFrogML will automatically save the result
        self.model = train_model(
            dataframe=dataframe,
            source_text=source,
            target_text=target,
            output_dir="outputs",
            model_params=self.model_params,
        )

    def schema(self):
        """
        JFrogML API Schema - Defines expected input format
        """
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),  # Financial question input
            ])

    def initialize_model(self):
        """
        JFrogML Runtime Initialization - Called once when model container starts
        
        This method runs once when JFrogML starts your model container for serving.
        Use this to:
        - Load tokenizers, preprocessors, or other dependencies
        - Set up device allocation (CPU/GPU)
        - Initialize any runtime-specific configurations
        
        This is separate from build() and only runs during serving, not training.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_params["model_id"],
            model_max_length=self.model_params["max_source_text_length"]
        )
        self.device = get_device()
        print(f"Inference using device: {self.device}")

    @frogml.api()
    def predict(self, df):
        """
        JFrogML Inference Phase - Called for each prediction request
        
        This method handles all incoming API requests when your model is deployed.
        The @frogml.api() decorator tells JFrogML this is the main prediction endpoint.
        
        Input: DataFrame with 'prompt' column (as defined in schema())
        Output: DataFrame with prediction results
        
        JFrogML automatically handles:
        - API request/response formatting
        - Batch processing
        - Load balancing and scaling
        - Monitoring and logging
        """
        # Tokenize the financial question
        input_ids = self.tokenizer(list(df['prompt'].values), return_tensors="pt").to(self.device)
        
        # Generate answer using fine-tuned T5 model
        outputs = self.model.generate(**input_ids, max_new_tokens=self.model_params["max_target_text_length"])
        
        # Decode the generated answer
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Return results in DataFrame format for JFrogML API
        return pd.DataFrame([{
            "generated_text": decoded_outputs
        }])