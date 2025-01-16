import tritonclient.grpc as grpcclient
from scipy.io import wavfile
import librosa
import numpy as np
import time

class InferenceTypeError(Exception):
    def __init__(self, message: str = "Inference Type must be 'grpc'"):
        self.message = message
        super().__init__(self.message)

class InferenceClient:
    def __init__(self, inference_type: str = "grpc"):
        if inference_type not in ["grpc"]:
            raise InferenceTypeError()

        self.inference_type = inference_type
        self.client, self.inference_client = self.configure_client()

    def configure_client(self):
        if self.inference_type == "grpc":
            return self._configure_grpc()

    def _configure_grpc(self):
        client = grpcclient
        host_url = "localhost:8001"
        inference_client = client.InferenceServerClient(url=host_url)
        return client, inference_client

    def make_input_for_whisper(self, audio_path):
        sampling_rate, audio_buffer = wavfile.read(audio_path)
        audio_buffer = audio_buffer.astype(np.float32)
        audio_buffer /= np.max(np.abs(audio_buffer))

        inputs = []
        inputs.append(self.client.InferInput("audio", [audio_buffer.shape[0],], datatype="FP32"))
        inputs[0].set_data_from_numpy(audio_buffer)

        inputs.append(self.client.InferInput("sampling_rate", [1], datatype="INT32"))
        inputs[1].set_data_from_numpy(np.array([sampling_rate]).astype(np.int32))

        return inputs

    def make_input_for_jina(self, text_input):
        # Prepare input for Jina model
        input_data = np.array([text_input.encode("utf-8")], dtype=np.object_)

        inputs = []
        inputs.append(self.client.InferInput("INPUT", [1], "BYTES"))  # Triton expects BYTES for TYPE_STRING
        inputs[0].set_data_from_numpy(input_data)

        return inputs

    def make_input_for_ko_sroberta(self, sentences):
        input_data = np.array([[sentence.encode("utf-8")] for sentence in sentences], dtype=np.object_)  # Shape: [batch_size, 1]

        inputs = []
        inputs.append(self.client.InferInput("INPUT", input_data.shape, "BYTES"))
        inputs[0].set_data_from_numpy(input_data)

        return inputs

    def make_input_for_nemo(self, segment):
        """
        Prepare a single audio segment for NeMo ASR model.
        """
        inputs = []
        inputs.append(self.client.InferInput("input_signal", [segment.shape[0]], datatype="FP32"))
        inputs[0].set_data_from_numpy(segment.astype(np.float32))
        return inputs

    def predict_for_nemo(self, audio_path, segment_length=5, sampling_rate=16000, triton_model_name="nemo_asr_model"):
        """
        Process audio file and send segments to Triton server.
        """
        duration = librosa.get_duration(filename=audio_path)
        segments = []

        for s in range(0, int(duration / segment_length) + 1):
            start = s * segment_length
            segment, _ = librosa.load(audio_path, offset=start, duration=segment_length, sr=sampling_rate)
            segments.append(segment)

        logits_list = []
        embs_list = []

        for segment in segments:
            inputs = self.make_input_for_nemo(segment=segment)

            # Request logits and embeddings
            logits = self.get_response(
                inputs=inputs,
                triton_model_name=triton_model_name,
                output_name="logits"
            )
            embs = self.get_response(
                inputs=inputs,
                triton_model_name=triton_model_name,
                output_name="embs"
            )

            logits_list.append(logits)
            embs_list.append(embs)

        return logits_list, embs_list

    def get_response(self, inputs, triton_model_name, output_name):
        server_response = self.inference_client.infer(model_name=triton_model_name, inputs=inputs)
        return server_response.as_numpy(name=output_name)

if __name__ == "__main__":
    model_type = input("Select model (1: whisper, 2: jina, 3: ko-sroberta, 4: nemo_asr): ").strip().lower()

    grpc_inference = InferenceClient(inference_type="grpc")

    start_time = 0.0
    if model_type == "1":
        audio_path = input("Enter audio file path (default: './test.wav'): ").strip()
        if audio_path == "": audio_path = "./test.wav"

        triton_model_name = "whisper-large-v3"
        output_name = "transcribed_text"

        start_time = time.time()
        inputs = grpc_inference.make_input_for_whisper(audio_path=audio_path)

        text = grpc_inference.get_response(
            inputs=inputs,
            triton_model_name=triton_model_name,
            output_name=output_name
        )
        decoded_text = "".join([t.decode("utf-8") for t in text])
        print("Transcription:", decoded_text)

    elif model_type == "2":
        text_input = input("Enter a sentence for Jina embedding: ").strip()
        triton_model_name = "jina-embeddings-v3"
        output_name = "OUTPUT"

        start_time = time.time()
        inputs = grpc_inference.make_input_for_jina(text_input=text_input)

        embeddings = grpc_inference.get_response(
            inputs=inputs,
            triton_model_name=triton_model_name,
            output_name=output_name
        )
        print("Embeddings shape:", embeddings.shape)
        print("Embeddings:", embeddings)

    elif model_type == "3":
        sentences = ["안녕하세요.", "이 문장은 테스트입니다.", "Triton 배치 추론 테스트."]
        triton_model_name = "ko-sroberta-multitask"
        output_name = "OUTPUT"

        start_time = time.time()
        inputs = grpc_inference.make_input_for_ko_sroberta(sentences)

        embeddings = grpc_inference.get_response(
            inputs=inputs,
            triton_model_name=triton_model_name,
            output_name=output_name
        )
        print("Embeddings shape:", embeddings.shape)
        print("Embeddings:", embeddings)

    elif model_type == "4":
        audio_path = input("Enter audio file path (default: './test.wav'): ").strip()
        if audio_path == "": audio_path = "./test.wav"

        segment_length_str = input("Enter segment length (sec) (default: 5): ").strip()
        if segment_length_str == "":
            segment_length = 5
        else :
            segment_length = int(segment_length_str)

        nemo_model_type  = input("Enter Triton model name:\n 1. nemo-asr-task1\n 2. nemo-asr-task3\n").strip()

        if nemo_model_type == "1":
            triton_model_name = "nemo-asr-task1"
        elif nemo_model_type == "2":
            triton_model_name = "nemo-asr-task3"

        start_time = time.time()

        # Perform segment-based prediction
        logits_list, embs_list = grpc_inference.predict_for_nemo(
            audio_path=audio_path,
            segment_length=5,  # Duration of each segment in seconds
            triton_model_name=triton_model_name
        )

        for i, (logits, embs) in enumerate(zip(logits_list, embs_list)):
            print(f"Segment {i}:")
            print(f"  Logits : {logits}")
            print(f"  Embeddings : {embs}")

        end_time = time.time()
        print(f"Inference Time: {end_time - start_time:.2f} seconds")

    else:
        print("Invalid model type selected.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Inference Time: {execution_time:.2f} seconds")
