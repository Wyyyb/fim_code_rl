#!/usr/bin/env bash
set -euo pipefail

# 单模型评估脚本：在 SWE-Bench-Verified 和 SWE-Bench-Lite 上评估一个模型
# 使用 2 块 GPU，每个数据集一块 GPU

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

#############################################
# 用户配置区域 - 直接在这里修改
#############################################

# 模型路径
MODEL_PATH="ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808"

# GPU 分配
GPU_VERIFIED=0      # SWE-Bench-Verified 使用的 GPU
GPU_LITE=1          # SWE-Bench-Lite 使用的 GPU

# 端口分配
PORT_VERIFIED=8400
PORT_LITE=8410

#############################################
# 其他配置
#############################################

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MAX_CONTEXT_LEN=65536
GPU_MEM_UTIL=0.90
SERVER_READY_TIMEOUT=3600

MAX_WORKERS=24
MAX_STEPS=40
MAX_STEPS_ABS=100
MAX_REWARD_CALC_TIME=1200

USE_EXISTING=True
TEMP=1
HF_OVERRIDES="{\"max_position_embeddings\": ${MAX_CONTEXT_LEN}}"

DATASET_VERIFIED="R2E-Gym/SWE-Bench-Verified"
DATASET_LITE="R2E-Gym/SWE-Bench-Lite"
SPLIT_VERIFIED="test"
SPLIT_LITE="test"
K_VERIFIED=500
K_LITE=300

TRAJ_BASE="./traj_swe_single_model"
RUN_LOG_DIR="./run_logs"
RESULTS_DIR="./results"
mkdir -p "${TRAJ_BASE}" "${RUN_LOG_DIR}" "${RESULTS_DIR}"

#############################################
# 函数定义
#############################################

docker_login_if_configured() {
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi
  local user="${DOCKERHUB_USERNAME:-${DOCKER_USERNAME:-}}"
  local pass="${DOCKERHUB_PASSWORD:-${DOCKER_PASSWORD:-}}"
  local token="${DOCKERHUB_TOKEN:-${DOCKER_TOKEN:-}}"
  local registry="${DOCKER_REGISTRY:-docker.io}"
  if [[ -z "${user}" || ( -z "${pass}" && -z "${token}" ) ]]; then
    return 0
  fi
  echo "docker login -> ${registry} (user=${user})"
  if [[ -n "${token}" ]]; then
    printf '%s' "${token}" | docker login "${registry}" -u "${user}" --password-stdin >/dev/null
  else
    printf '%s' "${pass}" | docker login "${registry}" -u "${user}" --password-stdin >/dev/null
  fi
  echo "docker login ok."
}

wait_for_server() {
  local port="$1"
  local pid="$2"
  local label="$3"
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "[${label}] vLLM exited early (pid=${pid}). Check logs in ${RUN_LOG_DIR}."
      return 1
    fi
    if curl -sSf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "[${label}] vLLM did not become ready in time (port=${port})."
  return 1
}

start_vllm_server() {
  local model_dir="$1"
  local served_name="$2"
  local gpu="$3"
  local port="$4"
  local label="$5"
  local log_file="${RUN_LOG_DIR}/vllm_${label}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  python -m vllm.entrypoints.openai.api_server \
    --model "${model_dir}" \
    --served-model-name "${served_name}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_CONTEXT_LEN}" \
    --hf-overrides "${HF_OVERRIDES}" \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --disable-log-requests \
    > "${log_file}" 2>&1 &
  STARTED_PID="$!"
}

launch_eval() {
  local served_name="$1"
  local dataset="$2"
  local split="$3"
  local traj_dir="$4"
  local exp_name="$5"
  local k="$6"
  local port="$7"
  local log_file="${RUN_LOG_DIR}/${exp_name}.log"

  (
    export LLM_BASE_URL="http://127.0.0.1:${port}/v1"
    export CUDA_VISIBLE_DEVICES=""
    python src/r2egym/agenthub/run/edit.py runagent_multiple \
      --traj_dir "${traj_dir}" \
      --max_workers "${MAX_WORKERS}" \
      --start_idx 0 \
      --k "${k}" \
      --dataset "${dataset}" \
      --split "${split}" \
      --llm_name "openai/${served_name}" \
      --scaffold "r2egym" \
      --use_fn_calling False \
      --use_existing "${USE_EXISTING}" \
      --exp_name "${exp_name}" \
      --temperature "${TEMP}" \
      --max_steps "${MAX_STEPS}" \
      --max_steps_absolute "${MAX_STEPS_ABS}" \
      --backend "docker" \
      --max_reward_calc_time "${MAX_REWARD_CALC_TIME}" \
      --max_tokens "${MAX_CONTEXT_LEN}" \
    > "${log_file}" 2>&1
  ) &
  LAUNCHED_PID="$!"
}

resolve_model_dir() {
  local root="$1"
  if [[ -f "${root}/config.json" ]]; then
    echo "${root}"
    return 0
  fi

  local candidates=()
  local cfg
  while IFS= read -r -d '' cfg; do
    local d
    d="$(dirname "${cfg}")"
    if { ls "${d}"/*.safetensors >/dev/null 2>&1 || ls "${d}"/*.bin >/dev/null 2>&1; }; then
      candidates+=("${d}")
    fi
  done < <(find "${root}" -maxdepth 6 -type f -name config.json -print0 2>/dev/null || true)

  if (( ${#candidates[@]} == 0 )); then
    echo "${root}"
    return 0
  fi

  local best="${candidates[0]}"
  local best_step=-1
  local d step base
  for d in "${candidates[@]}"; do
    base="$(basename "${d}")"
    if [[ "${base}" =~ ^checkpoint-([0-9]+)$ ]]; then
      step="${BASH_REMATCH[1]}"
    else
      step=-1
    fi
    if (( step > best_step )); then
      best_step="${step}"
      best="${d}"
    elif (( step == best_step )); then
      if [[ "${d}" > "${best}" ]]; then
        best="${d}"
      fi
    fi
  done

  if (( ${#candidates[@]} > 1 )); then
    echo "WARN: ${root} has multiple config.json candidates; using: ${best}" >&2
  fi
  echo "${best}"
}

#############################################
# 主流程
#############################################

docker_login_if_configured

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

MODEL_DIR="$(resolve_model_dir "${MODEL_PATH}")"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "ERROR: Model dir missing config.json: ${MODEL_DIR}"
  exit 1
fi

SERVED_NAME="$(basename "${MODEL_PATH}")"

echo "====================================================================="
echo "Model: ${MODEL_PATH}"
echo "  model_dir: ${MODEL_DIR}"
echo "  served_name: ${SERVED_NAME}"
echo "GPU allocation:"
echo "  - Verified: GPU ${GPU_VERIFIED}, port ${PORT_VERIFIED}"
echo "  - Lite: GPU ${GPU_LITE}, port ${PORT_LITE}"
echo "Datasets:"
echo "  - Verified: ${DATASET_VERIFIED} (${SPLIT_VERIFIED}), k=${K_VERIFIED}"
echo "  - Lite: ${DATASET_LITE} (${SPLIT_LITE}), k=${K_LITE}"
echo "Settings: temp=${TEMP} | max_len=${MAX_CONTEXT_LEN} | workers=${MAX_WORKERS} | use_existing=${USE_EXISTING}"
echo "Traj base: ${TRAJ_BASE}"
echo "====================================================================="

server_pids=()
cleanup() {
  echo "Stopping vLLM servers..."
  for pid in "${server_pids[@]:-}"; do
    if [[ "${pid}" =~ ^[0-9]+$ ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}
trap cleanup EXIT INT TERM

# 启动两个 vLLM 服务器
echo "Starting 2 vLLM servers..."
start_vllm_server "${MODEL_DIR}" "${SERVED_NAME}" "${GPU_VERIFIED}" "${PORT_VERIFIED}" "${SERVED_NAME}_verified"
server_pids+=("${STARTED_PID}")
start_vllm_server "${MODEL_DIR}" "${SERVED_NAME}" "${GPU_LITE}" "${PORT_LITE}" "${SERVED_NAME}_lite"
server_pids+=("${STARTED_PID}")

echo "Waiting for servers to be ready..."
wait_for_server "${PORT_VERIFIED}" "${server_pids[0]}" "${SERVED_NAME}_verified"
wait_for_server "${PORT_LITE}" "${server_pids[1]}" "${SERVED_NAME}_lite"
echo "All servers ready."

# 定义实验名称和目录
exp_verified="${SERVED_NAME}_swebv_verified"
exp_lite="${SERVED_NAME}_swebench_lite"

traj_verified="${TRAJ_BASE}/${exp_verified}"; mkdir -p "${traj_verified}"
traj_lite="${TRAJ_BASE}/${exp_lite}"; mkdir -p "${traj_lite}"

# 启动两个评估任务
echo "Launching evaluations..."
eval_pids=()
launch_eval "${SERVED_NAME}" "${DATASET_VERIFIED}" "${SPLIT_VERIFIED}" "${traj_verified}" "${exp_verified}" "${K_VERIFIED}" "${PORT_VERIFIED}"
eval_pids+=("${LAUNCHED_PID}")
launch_eval "${SERVED_NAME}" "${DATASET_LITE}" "${SPLIT_LITE}" "${traj_lite}" "${exp_lite}" "${K_LITE}" "${PORT_LITE}"
eval_pids+=("${LAUNCHED_PID}")

echo "Waiting for evaluations to complete..."
for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

# 生成提交文件
echo "Generating submissions..."
any_failed=0
for exp in "${exp_verified}" "${exp_lite}"; do
  traj_dir="${TRAJ_BASE}/${exp}"
  traj_file="${traj_dir}/${exp}.jsonl"
  if [[ ! -f "${traj_file}" ]]; then
    echo "  - [${exp}] ERROR: trajectory file not found: ${traj_file}"
    any_failed=1
    continue
  fi
  traj_lines="$(wc -l < "${traj_file}" | tr -d ' ')"
  if (( traj_lines == 0 )); then
    echo "  - [${exp}] ERROR: trajectory file is empty."
    any_failed=1
    continue
  fi
  python src/r2egym/agenthub/trajectory/create_swebench_submission.py \
    --traj_file_path "${traj_file}" \
    --output_json_path "${RESULTS_DIR}/${exp}_submission.json"
  echo "  - ${RESULTS_DIR}/${exp}_submission.json (${traj_lines} trajectories)"
done

if (( any_failed != 0 )); then
  echo "Done (with errors)."
  exit 1
fi
echo "Done."