# Default example
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
