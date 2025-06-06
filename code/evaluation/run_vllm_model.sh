#!/usr/bin/env sh
set -e

LOG_DIR="./log"
mkdir -p "$LOG_DIR"

# 실행할 모델 목록
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-Nemo-Instruct-2407"
  "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
  "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  # "gpt-4o"
  # "gpt-3.5-turbo-0125"
)

DEVICE="0,1"

# 고정된 컨텍스트 타입
CONTEXT_TYPES="birth Nationality Summary"
CTX_CLEAN="birth_Nationality_Summary"  # 로그 이름용 고정 값

# 질문 유형 리스트
QUESTION_TYPES="cross temporal cultural fact"

for MODEL in "${MODELS[@]}"; do
  for QTYPE in $QUESTION_TYPES; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    MODEL_NAME_CLEAN=$(echo "$MODEL" | tr '/ ' '__')
    LOG_FILE_NAME="${MODEL_NAME_CLEAN}_${CTX_CLEAN}_${QTYPE}.log"

    echo "[${TIMESTAMP}] Running model=${MODEL}, question_type=${QTYPE}, context_types=\"${CONTEXT_TYPES}\"..."

    if python script.py \
         --model_name "${MODEL}" \
         --question_type "${QTYPE}" \
         --device_index "${DEVICE}" \
         --context_types ${CONTEXT_TYPES} \
       2>&1 | tee "${LOG_DIR}/${LOG_FILE_NAME}"; then
      echo "  → 성공: 로그 → ${LOG_DIR}/${LOG_FILE_NAME}"
    else
      echo "  **실패**: model=${MODEL}, question_type=${QTYPE} 실행 중 에러."
      echo "         로그 → ${LOG_DIR}/${LOG_FILE_NAME}"
    fi

    echo
  done
done

echo "모든 작업 완료."
