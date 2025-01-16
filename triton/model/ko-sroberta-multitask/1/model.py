from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class TritonPythonModel:
    def initialize(self, args):
        # Logger
        self.logger = pb_utils.Logger
        self.model_name = "jhgan/ko-sroberta-multitask"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.log_info("ko-sroberta-multitask Model loaded successfully.")

    def execute(self, requests):
        responses = []

        # Collect all input texts from requests
        all_sentences = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_texts = input_tensor.as_numpy()  # Get batch of texts
            decoded_texts = [text.decode("utf-8") for text in input_texts]
            all_sentences.extend(decoded_texts)

        self.logger.log_info(f"Received sentences: {all_sentences}")

        # Tokenize input sentences
        tokenized = self.tokenizer(
            all_sentences, padding=True, truncation=True, return_tensors="pt", max_length=128
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Perform inference
        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply mean pooling
        sentence_embeddings = mean_pooling(model_output, attention_mask).cpu().numpy()

        # Split embeddings to match individual requests
        start_idx = 0
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            batch_size = input_tensor.as_numpy().shape[0]
            self.logger.log_info(f"Batch size: {batch_size}")

            # Slice embeddings for the current request
            request_embeddings = sentence_embeddings[start_idx:start_idx + batch_size]
            start_idx += batch_size

            # Create response for the current request
            output_tensor = pb_utils.Tensor("OUTPUT", request_embeddings)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses
