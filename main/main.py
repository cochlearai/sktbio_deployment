import tritonclient.grpc as grpcclient
from scipy.io import wavfile
import librosa
import numpy as np
import time
import onnxruntime as ort

import glob
import os
import pickle
import ast
import soundfile as sf
import json

SEGMENT_LENGTH = 5
MODE = "all"
# MODE = "fold_best"

if MODE == "all":
    TASK1_NEMO_MODEL_VERSION = "11"
    TASK3_NEMO_MODEL_VERSION = "11"
    LLM_CLF_FOLD = "all"
elif MODE == "fold_best":
    TASK1_NEMO_MODEL_VERSION = "10"
    TASK3_NEMO_MODEL_VERSION = "3"
    LLM_CLF_FOLD = "all"
else:
    TASK1_NEMO_MODEL_VERSION = None
    TASK3_NEMO_MODEL_VERSION = None
    LLM_CLF_FOLD = None

clf_list_fold = "./llm_clf/task3_llm_1024_mean_SVC_en_fold_{}_answer_{}.pkl"

TASK1_BEST_FOLD = 10    # 0.842
TASK3_BEST_FOLD = 3     # 0.842

TARGET_ANSWERS_TASK3 = [0, 1, 2, 3, 4, 5, 6]
# TARGET_ANSWERS_TASK3 = [0, 1, 3, 5]
TARGET_LANGUAGE = "en" # kr은 쓰지 않음 (베스트 성능이 아님)

# THRESHOLD_TASK1 = 0.5
# THRESHOLD_TASK3_AUDIO = 0.5
# THRESHOLD_TASK3_LLM = 0.5
# THRESHOLD_TASK3_VOTING = 0.5

result_code = ["SAD", "NORMAL"]
language_models = {
    "kr": "ko-sroberta-multitask",
    "en": "jina-embeddings-v3"
}

# TODO: en 모델은 translation이 되어야 함
output_names = {
    "kr": "transcribed_text",
    "en": "transcribed_text"
}

TASK3_CLF_TYPE = "mean" # mean or sentence

rnn_model_path = os.path.join(".", "rnn", 'model_task3_logit_fold{}.onnx')

providers = ['CPUExecutionProvider'] # for cpu

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
        # note: batch input 수정
        input_data = np.array([t.encode("utf-8") for t in text_input], dtype=np.object_)

        inputs = []
        inputs.append(self.client.InferInput("INPUT", input_data.shape, "BYTES"))  # Triton expects BYTES for TYPE_STRING
        inputs[0].set_data_from_numpy(input_data)

        return inputs

    def make_input_for_ko_sroberta(self, sentences):
        # note: batch input 수정
        input_data = np.array([sentence.encode("utf-8") for sentence in sentences], dtype=np.object_)  # Shape: [batch_size, 1]

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

    def predict_for_nemo(self, audio_path, segment_length=5, sampling_rate=16000, triton_model_name="nemo_asr_model", model_version="11"):
        """
        Process audio file and send segments to Triton server.
        """
        duration = librosa.get_duration(filename=audio_path)
        segments = []

        for s in range(0, int(duration / segment_length) + 1):
            start = s * segment_length
            segment, _ = librosa.load(audio_path, offset=start, duration=segment_length, sr=sampling_rate)
            if len(segment) < segment_length*sampling_rate:
                break
            segments.append(segment)

        logits_list = []
        embs_list = []

        for segment in segments:
            inputs = self.make_input_for_nemo(segment=segment)

            # Request logits and embeddings
            logits = self.get_response(
                inputs=inputs,
                triton_model_name=triton_model_name,
                output_name="logits",
                model_version=model_version
            )
            embs = self.get_response(
                inputs=inputs,
                triton_model_name=triton_model_name,
                output_name="embs",
                model_version=model_version
            )

            logits_list.append(logits)
            embs_list.append(embs)
        # note: 순서 바꿈
        return logits_list, embs_list

    def get_response(self, inputs, triton_model_name, output_name, model_version="1"):
        server_response = self.inference_client.infer(model_name=triton_model_name, 
                                                      inputs=inputs, 
                                                      model_version=model_version)
        return server_response.as_numpy(name=output_name)

def change_sampling_rate(audio_files, target_sr=16000):
    """
    Change the sampling rate of audio files to the target sampling rate.

    Parameters:
    audio_files (list): List of paths to audio files.
    target_sr (int): Target sampling rate (default is 16000).
    """
    for audio_path in audio_files:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue

        try:
            # Load the audio file and get its current sampling rate
            audio, sr = librosa.load(audio_path, sr=None)

            if sr == target_sr:
                print(f"File {audio_path} already has the target sampling rate of {target_sr}. Skipping.")
                continue

            # Resample the audio to the target sampling rate
            print(f"Resampling {audio_path} from {sr} to {target_sr}...")
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Save the resampled audio back to the same file
            sf.write(audio_path, audio_resampled, target_sr)
            print(f"File {audio_path} resampled and saved successfully.")

        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict_speaker_model(audio_list, triton_model_name, threshold, aggregate="softmax", model_version="11", rnn_model_num='all'):
    logits_list = []
    logits_softmax_list = []
    for audio_path in sorted(audio_list):
        _, logits = grpc_inference.predict_for_nemo(
            audio_path=audio_path,
            segment_length=5,  # Duration of each segment in seconds
            triton_model_name=triton_model_name,
            model_version=model_version
        )
        # print(logits)
        logits_list+=logits
    
    # for softmax mean (task1)
        if aggregate == "softmax":
            # 모든 logit 값에 대해 softmax를 취함
            if len(logits):
                logits_softmax = [list(softmax(list(logit[0]))) for logit in logits]
                logits_softmax_list.append(logits_softmax)
     
    # for softmax mean (task1)
    if aggregate == "softmax":
        print("logits:\n", logits_list)
        print("softmax logits:\n", logits_softmax_list)
        
        # 각 segment에 대해 평균을 취함 -> 각 답변당 하나의 softmax array
        logits_mean_list=[np.mean(l, axis=0) for l in logits_softmax_list]
        print("mean softmax logits for segments:\n", logits_mean_list)
        
        # 모든 답변에 대해 평균을 취함 -> 각 피험자당 하나의 softamx array
        total_average = list(np.mean(logits_mean_list, axis=0))
        print("total mean:\n", total_average)
        
        pred = total_average[0]
        prob_0 = total_average[0]
        prob_1 = 1-prob_0
    
    # for rnn (task3)
    elif aggregate == "rnn":
        logits_list = np.array(logits_list).reshape(-1, 2)
        print(logits_list.shape)
        
        ort_sess = ort.InferenceSession(rnn_model_path.format(rnn_model_num), providers=providers)
        inp_name = ort_sess.get_inputs()[0].name
        inp_shape = ort_sess.get_inputs()[0].shape # (batch, sequence, 2) 
        max_length = inp_shape[1] # RNN 모델의 시퀀스 길이 == 144
        
        if logits_list.shape[0] < max_length: 
            X_new = np.zeros((max_length,2))
            X_new[:logits_list.shape[0]] = logits_list
            logits_list = X_new
        elif logits_list.shape[0] > max_length:
            logits_list = logits_list[:max_length]
        print(logits_list.shape)
        # (sad_prob, norm_prob)
        pred_rnn = ort_sess.run(None, {ort_sess.get_inputs()[0].name: logits_list.reshape((1,logits_list.shape[0], logits_list.shape[1])).astype(np.float32)})

        pred = pred_rnn[0][0][0]
        prob_0 = pred_rnn[0][0][0]
        prob_1 = 1-prob_0
        
        '''
        # 모든 logit 값에 대해 softmax를 취함
        logits_list.append(logits)

        if len(logits):
            logits_softmax = [list(softmax(list(logit[0]))) for logit in logits]
            logits_softmax_list.append(logits_softmax)
        
            
    print("logits:\n", logits_list)
    print("softmax logits:\n", logits_softmax_list)
    
    # 각 segment에 대해 평균을 취함 -> 각 답변당 하나의 softmax array
    logits_mean_list=[np.mean(l, axis=0) for l in logits_softmax_list]
    print("mean softmax logits for segments:\n", logits_mean_list)
    
    # 모든 답변에 대해 평균을 취함 -> 각 피험자당 하나의 softamx array
    total_average = list(np.mean(logits_mean_list, axis=0))
    print("total mean:\n", total_average)
    
    # thresholding
    if total_average[0] >= threshold:
        predict = 0
    else:
        predict = 1
    '''
    
    # thresholding
    if pred > (1-threshold):
        predict = 0
    else:
        predict = 1
    
    return prob_0, prob_1, predict, logits_list, logits_softmax_list

def predict_speaker_model_logits(audio_path, triton_model_name):
    _, logits = grpc_inference.predict_for_nemo(
        audio_path=audio_path,
        segment_length=5,  # Duration of each segment in seconds
        triton_model_name=triton_model_name
    )
    print(logits)
    
    return np.array(logits)

import requests
import json

def send_diagnosis_callback(session_id, callback_url, diagnosis_results):
    """
    Sends diagnosis results to the callback API.

    Args:
        session_id (str): The session ID, used to build the user_seq in the callback URL.
        callback_url (str): The base callback URL.
        diagnosis_results (list): A list of diagnosis results containing task, code, and probability.

    Returns:
        dict: The response from the API.
    """
    # Extract user_seq from session_id 
    full_url = f"{callback_url}{session_id}"
    # print(full_url)

    # Prepare the headers and payload
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    payload = json.dumps(diagnosis_results)
    print(payload)

    # Make the PUT request
    try:
        response = requests.put(full_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

from flask import Flask, request, jsonify
import requests
import threading

app = Flask(__name__)
grpc_inference = InferenceClient(inference_type="grpc")


def process_tasks(data):

    a = time.time()

    print("############################## task 1 ##############################")

    # Sampling rate 
    
    base = '/mnt/audiofiles/'
    task1_audio_list = []
    for i in data['task1_files']:
        task1_audio_list.append(base+data['task1_files'][str(i)])

    change_sampling_rate (task1_audio_list)
    
    
    triton_model_name = "nemo-asr-task1"

    THRESHOLD_TASK1 = float(data["thresholds"]["THRESHOLD_TASK1"])
    THRESHOLD_TASK3_AUDIO = float(data["thresholds"]["THRESHOLD_TASK3_AUDIO"])
    THRESHOLD_TASK3_LLM = float(data["thresholds"]["THRESHOLD_TASK3_LLM"])
    THRESHOLD_TASK3_VOTING = float(data["thresholds"]["THRESHOLD_TASK3_VOTING"])

    if TASK1_NEMO_MODEL_VERSION != None:
        # audio model prediction
        task1_total_average0, task1_total_average1, predict_audio_task1, _, _ = predict_speaker_model(
            task1_audio_list,
            triton_model_name, 
            THRESHOLD_TASK1, 
            aggregate="softmax", 
            model_version=TASK1_NEMO_MODEL_VERSION
        )
        task1_total_average = [task1_total_average0, task1_total_average1]

    else:   # fold ensemble
        try:
            task1_total_average = [0.0, 0.0]
            task1_total_average_list = []
            # for fold in range(1, 11):
            #     # audio model prediction
            #     task1_total_average0, task1_total_average1, predict_audio_task1, _, logits_softmax_list = predict_speaker_model(
            #         task1_audio_list, 
            #         triton_model_name, 
            #         THRESHOLD_TASK1, 
            #         aggregate="softmax", 
            #         model_version=str(fold)
            #     )
            #     logits_mean_list=[np.mean(l, axis=0) for l in logits_softmax_list]
            #     total_average = list(np.mean(logits_mean_list, axis=0))
            #     # task1_total_average = [task1_total_average0, task1_total_average1]
            #     task1_total_average_list.append(total_average)
                
            # print(task1_total_average_list)
            # logit_fold_mean = np.mean(task1_total_average_list, axis=0)
            # print(logit_fold_mean)
            # pred = logit_fold_mean[0]
            # if pred > (1-THRESHOLD_TASK1):
            #     predict_audio_task1 = 0
            # else:
            #     predict_audio_task1 = 1
            
            # task1_total_average = [pred, 1-pred]

                # # audio 모델 최종 결과
                # print(f"Task 1 audio model 판정 결과: {result_code[predict_audio_task1]}")
                # print("\n\n")
        except Exception as e:
            print(e)
            
    # audio 모델 최종 결과
    # print(f"Task 1 audio model 판정 결과: {result_code[predict_audio_task1]}")
    # print("\n\n")
    
    print("############################## task 3 ##############################")
    
    if TASK3_NEMO_MODEL_VERSION != None:
        task3_total_agg = []
        
        task3_audio_list = []
        for i in data['task3_files']:
            task3_audio_list.append(base+data['task3_files'][str(i)])
        print("target3 audio files:\n", task3_audio_list)

        change_sampling_rate (task3_audio_list)
        
        triton_model_name = "nemo-asr-task3"
        temp1, temp2, predict_audio_task3, _, _ = predict_speaker_model(
            task3_audio_list, 
            triton_model_name, 
            THRESHOLD_TASK3_AUDIO, 
            aggregate="rnn", 
            model_version=TASK3_NEMO_MODEL_VERSION
        )
        task3_total_agg.append(predict_audio_task3)
        
        probs_spk = [temp1, temp2]
        
        # audio 모델 최종 결과
        print(f"Task 3 audio model 판정 결과: {result_code[predict_audio_task3]}")
        print(f"{probs_spk[predict_audio_task3]}")
        print("\n\n")

        # language model prediction
        language_model_preds = []
        for i, audio_path in enumerate(task3_audio_list):
            if i not in TARGET_ANSWERS_TASK3:
                continue
            print(f"answer num: {i}")
            language_models.get(TARGET_LANGUAGE)
            inputs = grpc_inference.make_input_for_whisper(audio_path=audio_path)
        
            transcribed_text = grpc_inference.get_response(
                inputs=inputs,
                triton_model_name="whisper-large-v3",
                output_name="transcribed_text",
                model_version="1"
            )
            
            transcribed_text = ast.literal_eval(ast.literal_eval(str(transcribed_text[0])).decode("utf-8"))
            
            text = transcribed_text['text']
            chunks = transcribed_text['chunks']
            
            print(chunks)
            
            decoded_text = "".join([t for t in text])
            print("decoded_text:\n", decoded_text)
            text_list = decoded_text.split(".")
            text_list = [t.strip().lstrip()+"." for t in text_list if t != '']
            print("Transcription:\n", text_list)

            llm_model_input = grpc_inference.make_input_for_jina(text_list)

            embeddings = grpc_inference.get_response(
                inputs=llm_model_input,
                triton_model_name=language_models.get(TARGET_LANGUAGE, "ko-sroberta-multitask"),
                output_name="OUTPUT"
            )
            
            # shape = (문장 수, 768 or 1024)
            print(embeddings.shape)
            print(embeddings)
            
            # sentence인지 mean인지에 따라서 feature matrix 생성
            feature_matrix = np.mean(embeddings, axis=0).reshape((1,-1))
            
            print(feature_matrix.shape)
            
            # pkl 파일에서 모델 불러오기
            clf_pkl = clf_list_fold.format(LLM_CLF_FOLD, i)
            
            with open(clf_pkl, "rb") as f:
                scaler, clf = pickle.load(f)
                
            # 추론
            print("feature_matrix:\n", feature_matrix)
            feature_matrix_scaled = scaler.transform(feature_matrix)
            print("feature_matrix_scaled:\n", feature_matrix_scaled)
            print(feature_matrix_scaled.shape)
            prob = clf.predict_proba(feature_matrix_scaled)        
            print("prob:\n", prob)
            
            language_model_preds+=list(np.argmax(prob, axis=1))
            
        print("language_model_preds:\n", language_model_preds)
        
        if np.sum(language_model_preds) > len(language_model_preds) * THRESHOLD_TASK3_LLM:
            predict_llm_task3 = 1
        else:
            predict_llm_task3 = 0
        
        prob_0_spk_task3 = temp1
        prob_0_llm_task3 = 1-np.sum(language_model_preds)/len(language_model_preds)
        probs_llm = [prob_0_llm_task3, 1-prob_0_llm_task3]
        task3_prob_0 = (prob_0_spk_task3+prob_0_llm_task3)/2
        task3_prob = [task3_prob_0, 1-task3_prob_0]
        
        print(f"Task 3 language model 판정 결과: {result_code[predict_llm_task3]}")
        print(f"{probs_llm[predict_llm_task3]}")
        
        # aggregate
        task3_total_agg.append(predict_llm_task3)
        
        if np.sum(task3_total_agg) > len(task3_total_agg) * THRESHOLD_TASK3_VOTING:
            predict_agg_task3 = 1
        else:
            predict_agg_task3 = 0
        
        print("predict_agg_task3:\n", task3_total_agg)
        print(f"Task 3 aggregate 판정 결과: {result_code[predict_agg_task3]}")
    
    else:   # fold_ensemble
        try:
            task3_total_agg = []
            
            # for audio model
            task3_audio_list = []
            for i in data['task3_files']:
                task3_audio_list.append(base+data['task3_files'][str(i)])
            print("target3 audio files:\n", task3_audio_list)
            
            triton_model_name = "nemo-asr-task3"
            audio_pred_list = []
            prob_sum = 0.
            for fold in range(1, 11):
                temp1, temp2, predict_audio_task3, _, _ = predict_speaker_model(
                    task3_audio_list, 
                    triton_model_name, 
                    THRESHOLD_TASK3_AUDIO, 
                    aggregate="rnn", 
                    model_version=str(fold),
                    rnn_model_num=str(fold)
                )
                print(temp1)
                prob_sum+=temp1
                audio_pred_list.append(np.argmax([temp1, temp2]))
            
            if np.mean(audio_pred_list) <= (1-THRESHOLD_TASK3_AUDIO):
                task3_total_agg.append(1)
            else:
                task3_total_agg.append(0)
            
            prob_0_spk = prob_sum/10
            print("prob_0_spk", prob_0_spk)
            
            # for llm_clf
            # language model prediction
            language_model_preds = []
            feature_matrix_list = []
            llm_answer_list = []
            for i, audio_path in enumerate(task3_audio_list):
                if i not in TARGET_ANSWERS_TASK3:
                    continue
                print(f"answer num: {i}")
                llm_answer_list.append(i)
                language_models.get(TARGET_LANGUAGE)
                inputs = grpc_inference.make_input_for_whisper(audio_path=audio_path)
            
                transcribed_text = grpc_inference.get_response(
                    inputs=inputs,
                    triton_model_name="whisper-large-v3",
                    output_name="transcribed_text",
                    model_version="1"
                )
                
                transcribed_text = ast.literal_eval(ast.literal_eval(str(transcribed_text[0])).decode("utf-8"))
                
                text = transcribed_text['text']
                chunks = transcribed_text['chunks']
                
                print(chunks)
                
                decoded_text = "".join([t for t in text])
                print("decoded_text:\n", decoded_text)
                text_list = decoded_text.split(".")
                text_list = [t.strip().lstrip()+"." for t in text_list if t != '']
                print("Transcription:\n", text_list)

                llm_model_input = grpc_inference.make_input_for_jina(text_list)

                embeddings = grpc_inference.get_response(
                    inputs=llm_model_input,
                    triton_model_name=language_models.get(TARGET_LANGUAGE),
                    output_name="OUTPUT"
                )
                
                # shape = (문장 수, 768 or 1024)
                print(embeddings.shape)
                print(embeddings)
                
                # sentence인지 mean인지에 따라서 feature matrix 생성
                feature_matrix = np.mean(embeddings, axis=0).reshape((1,-1))
                
                print("feature_matrix", feature_matrix)
                feature_matrix_list.append(feature_matrix)
            
            llm_prob_list_fold = []
            llm_preds = []
            for feature_index, feature_matrix in enumerate(feature_matrix_list):
                answer_num = llm_answer_list[feature_index]
                print(f"answer {answer_num}")

                llm_prob_list = []
                for i in range(1, 11):
                    print(f"fold {i}")
                    
                    # run sklearn model
                    with open(clf_list_fold.format(i, answer_num), "rb") as f:
                        scaler, clf = pickle.load(f)
                    
                    feature_matrix_scaled = scaler.transform(feature_matrix)
                    print(f"feature_matrix_scaled: {feature_matrix_scaled}")
                    prob = clf.predict_proba(feature_matrix_scaled)
                    print(f"prob: {prob}")
                    llm_prob_list.append(prob)
                    llm_prob_list_fold.append(prob)

                llm_prob_list = np.array(llm_prob_list)
                print(llm_prob_list.shape)
                llm_prob_list_mean = np.mean(llm_prob_list, axis=0)
                print(llm_prob_list_mean.shape)
                print(llm_prob_list_mean)
                llm_preds+=list(np.argmax(llm_prob_list_mean, axis=1))
                
            print(len(llm_preds))
            print(llm_preds)
            if np.sum(llm_preds) > len(llm_preds) * THRESHOLD_TASK3_LLM:
                task3_total_agg.append(1)
            else:
                task3_total_agg.append(0)
            
            prob_0_llm = 1-(np.sum(llm_preds)/len(llm_preds))
            print(np.mean(np.array(llm_prob_list_fold).reshape(-1, 2), axis=0))
            # 3. final vote
            if np.sum(task3_total_agg) > len(task3_total_agg) * THRESHOLD_TASK3_VOTING:
                predict_agg_task3 = 1
            else:
                predict_agg_task3 = 0
                
            task3_prob = [(prob_0_spk+prob_0_llm)/2, 1-(prob_0_spk+prob_0_llm)/2]
            
        except Exception as e:
            print(e)
        
    

    b = time.time()
    print(b-a)
    
    

    try:
        diagnosis_results = [
            {
                "task": "Task1",
                "prediction": result_code[predict_audio_task1],
                "probability": float(round(task1_total_average[predict_audio_task1],4)),
                "threshold_task1":THRESHOLD_TASK1
            },
            {
                "task": "Task3",
                "prediction": result_code[predict_agg_task3],
                "probability": float(round(task3_prob[predict_agg_task3],4)),
                "threshold_task3_audio":THRESHOLD_TASK3_AUDIO,
                "threshold_task3_llm":THRESHOLD_TASK3_LLM,
                "threshold_task3_voting":THRESHOLD_TASK3_VOTING
            }
        ]
        print(diagnosis_results)

        #################### make SAD score graph with transcribed text ####################

        splited = task3_audio_list[0].split('/')
        splited_drl = splited[3]
        json_dst = base + '/' + splited_drl

        print(json_dst + '$$$$$$$$$$$$$$$$$$$$$$$$$$')

        triton_model_name = "nemo-asr-task3"
        y_full = []
        print("make full audio")
        for i, audio_path in enumerate(task3_audio_list):
            y, sr = librosa.load(audio_path, sr=None)
            y_full += list(y)
        sf.write("temp.wav", np.array(y_full), sr)
        
        print("call to get logits")
        logits = predict_speaker_model_logits("temp.wav", triton_model_name).reshape(-1, 2)
        scores = logits[:, 0]
        print(scores)
        
        scores_json = []
        for i, s in enumerate(scores):
            scores_json.append({"time": (i+1)*5.0, "score": round(float(s), 4)})
        
        with open(json_dst+"/scores.json" ,"w") as f:
            json.dump(scores_json, f, indent=4)
        
        print("make input")
        inputs = grpc_inference.make_input_for_whisper(audio_path="temp.wav")
        
        print("call")
        transcribed_text = grpc_inference.get_response(
            inputs=inputs,
            triton_model_name="whisper-large-v3",
            output_name="transcribed_text",
            model_version="2"    # for korean
        )
        
        print("eval")
        transcribed_text = ast.literal_eval(ast.literal_eval(str(transcribed_text[0])).decode("utf-8"))
        
        text = transcribed_text['text']
        chunks = transcribed_text['chunks']
        
        text_json = []
        print(chunks)
        for i, c in enumerate(chunks):
            if c["text"].lstrip().rstrip() != "":
                text_json.append({"start_time": c['timestamp'][0], "end_time": c['timestamp'][1], "text": c["text"].lstrip().rstrip()})
        
        with open(json_dst+"/texts.json", "w", encoding="UTF-8") as f:
            json.dump(text_json, f, indent=4, ensure_ascii=False)

            
            
        # Send results to the callback URL

        # # Example usage
        # callback_data = {
        #     "session_id": data['session_id'],
        #     "callback": data['callback'],
        # }

        score_stt_results = {
            "scores": json_dst+"/scores.json",
            "texts": json_dst+"/texts.json"
        }
        
        response = send_diagnosis_callback(data["session_id"], data["callback"], diagnosis_results)
        # response_stt = send_diagnosis_callback(data["session_id"], data["callback"], score_stt_results)
        # print(response_stt)
    except Exception as e:
        print(e)

import threading
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.route('/sadvoice', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        required_keys = {"session_id", "callback", "task1_files", "task3_files","thresholds"}
        if not required_keys.issubset(data.keys()):
            return jsonify({"error": "Missing required keys in payload"}), 400

        try:
            executor.submit(process_tasks, data)
        except Exception as e:
            print(f"process_tasks errors: {e}")

        # process_tasks(data)

        return jsonify({"status": "Processing started"}), 200 

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
