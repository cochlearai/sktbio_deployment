import os
import numpy as np
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the Triton Python model.
        """
        model_path = "/mnt/raw_model/max_epoch_50_target_task3_fold_1_cur_epoch_30.nemo"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = EncDecSpeakerLabelModel.restore_from(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def execute(self, requests):
        responses = []

        for request in requests:
            input_signal = pb_utils.get_input_tensor_by_name(request, "input_signal").as_numpy()

            logits, embs = self.model.infer_segment(input_signal)  # input_signal만 사용

            logits_tensor = pb_utils.Tensor("logits", logits.cpu().numpy())
            embs_tensor = pb_utils.Tensor("embs", embs.cpu().numpy())

            responses.append(pb_utils.InferenceResponse(output_tensors=[logits_tensor, embs_tensor]))

        return responses

    def finalize(self):
        """
        Clean up resources when the Triton model is unloaded.
        """
        pass
