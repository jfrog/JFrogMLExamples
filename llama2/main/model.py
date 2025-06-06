import qwak
from qwak.model.schema import ModelSchema, ExplicitFeature
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from pandas import DataFrame
from qwak.model.base import QwakModel
from huggingface_hub import login
from huggingface_hub import snapshot_download
from qwak.clients.secret_service import SecretServiceClient


class Llama2MT(QwakModel):
    """The Model class inherit QwakModel base class"""

    def __init__(self):
        self.model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.model = None
        self.tokenizer = None

    def build(self):
        secret_service: SecretServiceClient = SecretServiceClient()
        hf_token = secret_service.get_secret("hf-blm-apikey")
        art_url = secret_service.get_secret("hf-blm-remote-repo")
        login(token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        snapshot_download(
            repo_id=self.model_id, etag_timeout=86400
        )

    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        secret_service: SecretServiceClient = SecretServiceClient()
        hf_token = secret_service.get_secret("hf-blm-apikey")
        art_url = secret_service.get_secret("hf-blm-remote-repo")
        login(token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(device=self.device, dtype=torch.bfloat16)

    @qwak.api()
    def predict(self, df):
        input_text = list(df['prompt'].values)
        input_ids = self.tokenizer(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=100)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pd.DataFrame([{"generated_text": decoded_outputs}])
