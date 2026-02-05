
import os
import base64

import boto3
from transformers import pipeline

import frogml
from sagemaker.serve.spec.inference_spec import InferenceSpec
from . import config

class DevopsAssistantInferenceSpec(InferenceSpec):
    """Custom InferenceSpec for HuggingFace text generation models."""
    
    def __init__(self):
        self.model_version = os.environ.get("MODEL_VERSION")

    def _get_secret_id(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"{name} is not set")
        return value

    def _get_secret_value(self, secret_id: str) -> str:
        client = boto3.client("secretsmanager", region_name="us-east-1")
        response = client.get_secret_value(SecretId=secret_id)
        secret = response.get("SecretString")
        if secret is None:
            secret = base64.b64decode(response["SecretBinary"]).decode("utf-8")
        return secret
    
    def load(self, model_dir: str):
        """Load HuggingFace model and tokenizer."""

        jf_token_secret_id = self._get_secret_id("JF_ACCESS_TOKEN_SECRET_ID")
        os.environ["JF_ACCESS_TOKEN"] = self._get_secret_value(jf_token_secret_id)

        print(f"Loading model {config.JF_MODEL_NAME} from {config.JF_REPO} version {self.model_version}")
        model, tokenizer = frogml.huggingface.load_model(
            model_name=config.JF_MODEL_NAME,
            repository=config.JF_REPO,
            version=self.model_version,
        )

        if model is None or tokenizer is None:
            raise ValueError(
                f"Failed to load {self.model_name} from {self.model_repository} "
                f"(version={self.model_version or 'latest'})"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        return {"generator": generator}
    
    def invoke(self, input_object, model):
        """Generate text using the HuggingFace model."""
        # Real HuggingFace inference
        if isinstance(input_object, dict) and "inputs" in input_object:
            text = input_object["inputs"]
            parameters = input_object.get("parameters", {})
        else:
            text = str(input_object)
            parameters = {}

        generator = model["generator"]
        defaults = {
            "max_new_tokens": 30,
            # Avoid echoing the prompt in the response by default.
            "return_full_text": False,
        }
        gen_args = {**defaults, **parameters}

        outputs = generator(text, **gen_args)
        return outputs

print("HuggingFace InferenceSpec defined successfully!")