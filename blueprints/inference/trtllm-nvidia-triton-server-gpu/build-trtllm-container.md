> NOTE: Build container for each GPU architecture (like Ada Lovelace, Hopper, Ampere, etc)

# Step 0: Setup environment
```bash
# Install Miniconda (if not already installed)
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

# Initialize conda for all shells and create a new environment
conda init --all
conda create -n ai-on-eks python=3.12
conda activate ai-on-eks

# Navigate to the project directory and install required Python dependencies
cd ~/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu
pip install -r requirements.txt
```

# Step 1: Setup tensorrt-backend
```bash
# Clone the Triton TensorRT-LLM backend repo
cd ~
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.17.0
cd tensorrtllm_backend

# Install Git LFS (used to fetch large files like model weights)
sudo apt update
sudo apt-get install git-lfs -y --no-install-recommends
git lfs install

# Download all submodules (required for the backend to function properly)
git submodule update --init --recursive
```

# Step 2: Download model weights
```bash
# Create a directory to store model weights and download LLaMA model using Hugging Face CLI
cd ~/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu 
mkdir llama
cd llama
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir Llama-3.2-1B-Instruct
```

# Step 3: Start Docker (triton server)
```bash
# Create directories to mount into the Docker container for engines and model files
cd ~/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu
mkdir triton_engines
mkdir triton_model_files

# Start the Triton inference server in interactive mode
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus=1 \
    -v /home/ubuntu/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu/llama:/llama \
    -v /home/ubuntu/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu/triton_engines:/engines \
    -v /home/ubuntu/tensorrtllm_backend:/tensorrtllm_backend \
    -v /home/ubuntu/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu/triton_model_files:/triton_model_files \
    nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3
```

> Note: Run the below steps from inside the container until "Stop the server"

# Step 4: Build the TensorRT model

```bash
# Define paths and convert Hugging Face model to TensorRT-LLM checkpoint
HF_LLAMA_MODEL=/llama/Llama-3.2-1B-Instruct
TENSORRTLLM_CKPT_PATH=/tmp/tensorrt_checkpoint/llama32/1b/
ENGINE_DIR=/engines
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py

# Convert HF model to TRTLLM checkpoint format
python ${CONVERT_CHKPT_SCRIPT} \
--model_dir ${HF_LLAMA_MODEL} \
--output_dir ${TENSORRTLLM_CKPT_PATH} \
--dtype float16 \
--tp_size 1

# Build TensorRT engine from the converted checkpoint
trtllm-build --checkpoint_dir ${TENSORRTLLM_CKPT_PATH} \
--output_dir ${ENGINE_DIR} \
--gemm_plugin auto \
--kv_cache_type paged \
--multiple_profiles enable \
--use_paged_context_fmha enable \
--max_batch_size 128 \
--max_num_tokens 65536
```

# Step 5: Set values in model files
```bash
# Copy the default inflight_batcher_llm model structure to Triton model directory
cp -r /tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_model_files/

# Define common environment variables to be passed to fill_template script
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
MODEL_FOLDER=/triton_model_files
TOKENIZER_DIR=${HF_LLAMA_MODEL}
MAX_BATCH_SIZE=128
LOGITS_DT=TYPE_FP32
TOKENIZER_TYPE=auto
DECOUPLED_MODE=false
STREAMING_POSTPROCESS_TOKENS_TOGETHER=True
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=1000
TRITON_BACKEND=tensorrtllm
BATCHING_STRATEGY=inflight_fused_batching
KV_CACHE_REUSE=True
ENCODER_IPF_DT=TYPE_FP16
EXCLUDE_INPUT_IN_OUTPUT=TRUE
ENABLE_CHUNKED_CONTEXT=TRUE
ENABLE_CONTEXT_FMHA_FP32_ACCUMULATION=TRUE
SCHEDULER=max_utilization

# Fill configuration for each Triton model component using template script
python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:${LOGITS_DT}

python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}

python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}

python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},accumulate_tokens:${STREAMING_POSTPROCESS_TOKENS_TOGETHER},logits_datatype:${LOGITS_DT}

python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:${BATCHING_STRATEGY},logits_datatype:${LOGITS_DT},enable_kv_cache_reuse:${KV_CACHE_REUSE},encoder_input_features_data_type:${ENCODER_IPF_DT},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},enable_context_fmha_fp32_acc:${ENABLE_CONTEXT_FMHA_FP32_ACCUMULATION},batch_scheduler_policy:${SCHEDULER}
```

# Step 6: Start the server from inside container
```bash
# Launch Triton inference server using the configured model repository
python /tensorrtllm_backend/scripts/launch_triton_server.py \
--world_size=1 \
--model_repo=${MODEL_FOLDER}
```

# Step 7: Run Inference/test model (in a new terminal)
```bash
# Send a sample generation request to Triton server
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Explain in simple words, the following concepts: reward modeling, policy distillation, offline RL loops, and when to ditch RL for direct‑pref learning.", "max_tokens": 500, "bad_words": "", "stop_words": ""}'
```

# Step 8: Stop the server (from original container terminal)
```bash
# Stop the running Triton server
pkill tritonserver

# Exit the Docker container
exit
```

# Step 9: Build the Docker container
> Note: At present, you should have successfully exited the container
```bash
# Clean up any MacOS-specific artifacts and build the Docker image
cd ~/ai-on-eks/blueprints/inference/trtllm-nvidia-triton-server-gpu
rm -rf __MACOSX
docker build -t triton-trtllm:latest .
```

# Step 10: Run and test
```bash
# Run the built container and expose Triton server port
docker run -d --runtime=nvidia --gpus all -p 8000:8000 triton-trtllm:latest

# Run a basic test query
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'

# List running containers and stop the one we just started
docker ps -a
docker kill <container-id>
```

# Step 11: Push to Amazon ECR
> NOTE: Ensure the IAM Role has permissions to create ecr repo(AmazonEC2ContainerRegistryFullAccess)
```bash
aws configure

# Create a new ECR repository (run only once)
aws ecr create-repository --repository-name triton-trtllm

# Authenticate Docker with ECR registry
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <replace-with-account-id>.dkr.ecr.<replace-with-region-id>.amazonaws.com

# Tag the image to match your ECR repo format
docker tag triton-trtllm:latest <replace-with-account-id>.dkr.ecr.<replace-with-region-id>.amazonaws.com/triton-trtllm:latest

# Push the image to your ECR repository
docker push <replace-with-account-id>.dkr.ecr.<replace-with-region-id>.amazonaws.com/triton-trtllm:latest
```