# run ollma in gpu mode
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# pull qwen2:0.5b
curl http://localhost:11434/api/pull -d '{
  "name": "llama3.1:8b"
}'

# run inference
# curl http://localhost:11434/api/generate -d '{
#   "model": "qwen2:0.5b",
#   "prompt": "Why is the sky blue?",
#   "stream": false
# }'