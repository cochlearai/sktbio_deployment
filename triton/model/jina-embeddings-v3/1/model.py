import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer
import time


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the Triton Python model.
        """
        self.logger = pb_utils.Logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.log_info("Loading tokenizer...")
        # self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

        self.logger.log_info("Loading model...")
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(self.device)

        self.logger.log_info("Model initialized successfully.")

    def execute(self, requests):
        """
        Execute inference on the model.
        """
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            start_time = time.perf_counter()
            # Retrieve input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()

            # Decode input
            texts = [x.decode("utf-8") for x in input_data]
            self.logger.log_info(f"Decoded texts: {texts}")

            # Tokenize input
            # inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Perform inference
            # embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
            embeddings = self.model.encode(texts, task="classification", truncate_dim=1024)
            self.logger.log_info(f"Embeddings shape: {embeddings.shape}")

            # Convert output
            # embeddings = embeddings.to(dtype=torch.float32)
            # output_tensor = pb_utils.Tensor("OUTPUT", embeddings.cpu().numpy())
            output_tensor = pb_utils.Tensor("OUTPUT", embeddings)

            self.logger.log_info(f"Time taken for single request: {time.perf_counter()- start_time}")
            # Append response
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        self.logger.log_info(f"Time taken by batch: {time.perf_counter()- start_time_batch}")
        return responses

    def finalize(self):
        """
        Finalize the Triton Python model.
        """
        self.logger.log_info("Finalizing model.")
