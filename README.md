# sktbio_deployment

## 1. 서버 환경 구성 과정
서버 환경 구성은 cochl-gpu 인스턴스에 이미 구성이 되어있으나, 다른 환경에 다시 구성을 하실 경우에 대비에서 간략하게 설명을 남겨 놓도록 하겠습니다. 
본 서버 리소스는 AI 모델들을 서빙하는 Triton Inference Server와 웨이센과의 연결을 위해 구성된 API 서버의 역할을 하는 Flask Server (Main App)으로 이루어져 있습니다. 
본 설명은 Azure VM에 GPU 셋업을 마친 상황에 전제하여 설명됩니다.

### (1) Main Server (Flask)
Main App의 리소스는 main 폴더 안에 담겨져 있으며, Python3 가상환경 (venv)를 구성하신 후에 requirements.txt에 정의되어 있는 라이브러리들을 설치하시면 바로 구동이 가능합니다. 

Venv 환경 구동 후 라이브러리 설치:
```sh
./build_triton.sh
./run_triton.sh
```

### (2) Triton Server
> Forked from https://github.com/sids07/Deployment_using_nvidia_triton_server/tree/main

This is a test repository for deploying multi-models on triton server

Current supported model
- [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- [ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)
- nemo-asr-task1 (our model)
- nemo-asr-task3 (our model)

### How to use

Run:
```sh
./build_triton.sh
./run_triton.sh
```

Check logs:
```sh
./log_triton.sh
```

Stop:
```sh
./stop_triton.sh
```
