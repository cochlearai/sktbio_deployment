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
$ (venv) $ pip3 install -r requirements.txt
```

설치 후 main server 구동: 
```sh
$ (venv) $ python main.py
```

Main Server의 구동까지 테스트 해보셨다면 다음으로는 정의된 flask app을 gunicorn을 통해서 expose 하는 작업입니다. Gunicorn을 통해 서버를 expose 하는 방식은 본 [링크](https://velog.io/@jiyoung/GunicornNginx-%EB%A6%AC%EB%88%85%EC%8A%A4-%EC%84%9C%EB%B2%84%EC%97%90%EC%84%9C-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0) 를 통해서 한번 익혀보시기 바랍니다. 

Gunicorn Service 정의: 
```sh
cd /etc/systemd/system
vim flask-app.service
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

Service 실행/등록/중단:
```sh
$ (venv) $ sudo systemctl start flask-app.service
$ (venv) $ sudo systemctl enable flask-app.service
$ (venv) $ sudo systemctl stop flask-app.service
```

Service 확인:
```sh
$ (venv) $ sudo systemctl status flask-app.service
```

위의 과정까지 마치면 Main Server를 구동할 환경 설정을 마치신 것으로 보아도 됩니다.

### (2) Triton Server
> Forked from https://github.com/sids07/Deployment_using_nvidia_triton_server/tree/main

Triton Server는 Dockerize하여 제공해드립니다. Docker가 설치되어 있으신 전제하에서는 sh 파일 실행을 통해서 쉽게 triton server를 띄우실 수 있게 준비해두었습니다. 사용 방식은 아래에 설명해두었습니다.

현재 제공하고 있는 모델 목록
- [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- [ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)
- nemo-asr-task1 (our model, 11개)
- nemo-asr-task3 (our model, 11개)

### 사용 방법

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

## 2. Main Server 구동 및 테스트 방법
전 파트에서 Main Server와 Triton Server의 환경 구성을 하시고 구동하는 방법을 보셨다면 이 파트에서는 서버를 구동하고 여러 가지 환경에서 이를 테스트 하시는 방법에 대해서 설명드립니다. 
이번 파트부터는 cochl-gpu안에 구성되어 있는 환경 기준으로 설명드립니다. 이에 cochl-gpu vm에 접속하셔서 그대로 실행하시면 된다는 말을 의미합니다. 

### (1) Main Server 구동 및 기타 Tip

Main Server 실행:
```sh
cd sblee-cochl/work/main/
source bin/activate
(main) sblee-cochl@cochl-server-gpu:~/work/main$ cd src/deployment
(main) sblee-cochl@cochl-server-gpu:~/work/main/src/deployment$ python main.py
```

그렇게 되면 다음과 같은 메시지를 보실 수 있으실 겁니다. 
```sh
 * Serving Flask app 'main'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:9000
 * Running on http://11.0.0.7:9000
Press CTRL+C to quit
```

위의 메시지를 보시면 서버가 9000 port 열려 있는 것을 보실 수 있으실 겁니다. 현재 5000 port는 flask-app.service가 이미 쓰고 있어서 따로 테스트용 server를 구동하실때는 같은 port를 쓰실 수 없습니다. 
테스트 용도로 따로 돌려보실 때는 9000 혹은 다른 port로 flask app을 구동하시고, 변경 사항을 flask-app.service에 적용하실 때는 5000으로 꼭 지정해주시기 바랍니다. Port 정보를 바꾸는 방법은 다음과 같습니다.

main.py의 817 line
```sh
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
```
위의 port를 9000 혹은 5000으로 바꿔가면서 띄어주세요.

### (2) Client.py 실행 방식

Main Server를 구동하셨으면, Client를 통해서 이를 테스트 해보실 차례입니다. 같은 디렉토리의 client.py를 실행하시면됩니다. 
그러나 client.py에는 localhost:9000으로 일단 테스트 해보실 수 있게 설정되있지만 테스트 할려고 하시는 url에 맞게 사용하시면 됩니다. 
API request를 하는 body에 대한 설명을 드리도록 하겠습니다.

API Request Body:
```sh
data = {
    "session_id": "1",
    "callback": "https://dev.mentai.waymed.ai:18040/api/v1/diagnosis/",
    "task1_files": {
        "1": "test202501131257/1-1.wav",
        "2": "test202501131257/1-2.wav",
        "3": "test202501131257/1-3.wav",
        "4": "test202501131257/1-4.wav",
        "5": "test202501131257/1-5.wav",
        "6": "test202501131257/1-6.wav",
        "7": "test202501131257/1-7.wav",
        "8": "test202501131257/1-8.wav",
        "9": "test202501131257/1-9.wav"
    },
    "task3_files": {
        "1": "test202501131257/3-1.wav",
        "2": "test202501131257/3-2.wav",
        "3": "test202501131257/3-3.wav",
        "4": "test202501131257/3-4.wav",
        "5": "test202501131257/3-5.wav",
        "6": "test202501131257/3-6.wav",
        "7": "test202501131257/3-7.wav"
    },
    "thresholds": {
        "THRESHOLD_TASK1":"0.5",
        "THRESHOLD_TASK3_AUDIO":"0.5",
        "THRESHOLD_TASK3_LLM":"0.5",
        "THRESHOLD_TASK3_VOTING":"0.5"
    }
}
```
다음과 같이 되어있는데 "session_id"는 웨이센에서 정의해서 보내주실 예정에 있기 때문에 그냥 테스트용도를 위해서는 "1"로 계속 테스트하셔도 될듯합니다. 
가장 중요한 것은 "task1_files"와 "task3_files"에 정의되는 오디오 파일들의 위치인데 테스트를 위해서는 이것은 특정 위치에 폴더를 만드셔서 오디오 파일을 넣어주셔야 한다는 것을 의미합니다.
현재 client.py에 정의되어있는 test202501131257는 이상민 연구원과 제가 값 비교를 위해서 같이 쓰고 있는 폴더이니 이것을 그냥 쓰셔서 테스트 해보셔도 됩니다. 

Thresholds는 김창현 매니저님이 요청 주신 4개의 threshold값을 API Request 할때 던져줄 수 있도록 정의되어 있습니다. 테스트 하실 때 테스트하시려고 하는 threshold 값을 넣어주면서 테스트해보시길 바랍니다.

새로운 오디오 파일을 추가하는 방법과 디테일에 대해서는 계속 아래에 설명드리도록 하겠습니다.

#### 테스트할 오디오 파일 추가 하기
웨이센에서 API Request를 보내기 위해서는 (1) 오디오 파일을 shared folder에 추가하기 (2) API Request 보내기의 순서로 이루어 지는데 shared folder는 Azure에서 제공하는 Shared Files 기능을 사용하기 때문에 본 서버는 다른 방식으로는 동작하지 않습니다. 
반드시 알려드리는 위치에 오디오 파일을 넣어주시고 테스트를 해봐주셔야 합니다. 

오디오 파일 업로드 위치:
```sh
/mnt/audiofiles/폴더명/
```
반드시 올리시려는 세션의 폴더 이름을 정의해주신 이후에 그 안에 오디오 파일을 넣어주셔야 합니다. 본 API의 request를 정상적으로 하기 위해서는 총 16개의 오디오 파일이 필요하며, 각 파일의 번호 Task 1의 경우 1번 문장 2번 문장...Task 3의 경우 1번 답변 2번 답변... 식으로 지정을 꼭 해주셔야 합니다. 해보시고 추가적으로 질문 있으시면 연락 주시기 바랍니다.

### (3) 기타 패러미터 변경 방법
기타 패러미터 변경 방법에 대해서 문의 주셨습니다. 이에 대해서 변경 방법에 대해서 알려드립니다. 

#### 모델 버전 설정 방법
특정 모델을 쓰고 싶은 경우에는 다음과 같이 이를 조정하실 수 있으십니다. MODE = "all"은 183명의 데이터를 다 써서 학습한 all model을 사용하는 것이고, 실험 사항의 10 fold 모델을 각각 사용해보시려면 MODE를 "fold_best"로 바꾸시고 1~10 사이의 숫자를 기입해주시면 됩니다. 

main.py line 16부터:
```sh
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
```

#### Task 3 특정 답변 만을 쓰고 싶은 경우
Task 3의 특정 답변만을 써서 평가하고 싶으신 경우에는 다음과 같이 조정하실 수 있으십니다. 0번 답변부터 6번 답변까지 리스트의 형태로 쓰고 싶으신 것만 선언해서 쓰시면 됩니다.

main.py line 37부터:
```sh
TARGET_ANSWERS_TASK3 = [0, 1, 2, 3, 4, 5, 6]
# TARGET_ANSWERS_TASK3 = [0, 1, 3, 5]
TARGET_LANGUAGE = "en" # kr은 쓰지 않음 (베스트 성능이 아님)
```

아래의 패러미터 변경을 하신 이후에는 저장하고 main.py를 다시 띄우시면 이 것이 반영이 되고 웨이센의 서버에 이를 반영하고 싶으시면 서비스를 중지하신 후에 재등록하고 다시 실행하시면 됩니다.

감사합니다.
