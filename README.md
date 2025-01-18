# sktbio_deployment

## 1. 서버 환경 구성 과정
서버 환경 구성은 cochl-gpu 인스턴스에 이미 구성이 되어있으나, 다른 환경에 다시 구성을 하실 경우에 대비에서 간략하게 설명을 남겨 놓도록 하겠습니다. 
본 서버 리소스는 AI 모델들을 서빙하는 Triton Inference Server와 웨이센과의 연결을 위해 구성된 API 서버의 역할을 하는 Flask Server (Main App)으로 이루어져 있습니다. 
본 설명은 Azure VM에 GPU 셋업을 마친 상황에 전제하여 설명됩니다.

### (1) Main Server (Flask)
Main App의 리소스는 main 폴더 안에 담겨져 있으며, Python3 가상환경 (venv)를 구성하신 후에 requirements.txt에 정의되어 있는 라이브러리들을 설치하시면 바로 구동이 가능합니다. 

Venv 환경 구동 후 라이브러리 설치:
```sh
source bin/activate (만드신 venv 프로젝트 폴더에 들어가신 후)
pip3 install -r requirements.txt
```

설치 후 main server 구동: 
```sh
python main.py
```

Main Server의 구동까지 테스트 해보셨다면 다음으로는 정의된 flask app을 gunicorn을 통해서 expose 하는 작업입니다. Gunicorn을 통해 서버를 expose 하는 방식은 본 [링크](https://velog.io/@jiyoung/GunicornNginx-%EB%A6%AC%EB%88%85%EC%8A%A4-%EC%84%9C%EB%B2%84%EC%97%90%EC%84%9C-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0) 를 통해서 한번 익혀보시기 바랍니다. 

Gunicorn Service 정의: 
```sh
cd /etc/systemd/system
vim flask-app.service
```

flask-app.service 안 내용 정의
```sh
[Unit]
Description=Flask Application
After=network.target

[Service]
User=sblee-cochl
WorkingDirectory=/home/sblee-cochl/work/main/src/deployment
ExecStart= /home/sblee-cochl/work/main/bin/gunicorn --workers 3 --bind 0.0.>

[Install]
WantedBy=multi-user.target
```

Gunicron Service 시작 / 중단:
```sh
$ (venv) $ sudo systemctl start gunicorn
$ (venv) $ sudo systemctl enable gunicorn
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
