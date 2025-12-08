from google import genai
import torch
import faiss
torch.cuda.is_available()
client = genai.Client()
model_info = client.models.get(model="gemma-3n-e2b-it")
print(f"{model_info.input_token_limit=}")
print(f"{model_info.output_token_limit=}")
print(faiss.__version__)