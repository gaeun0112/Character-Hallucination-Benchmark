#!/usr/bin/env sh
set -e

LOG_DIR="./log"
mkdir -p "$LOG_DIR"

INPUT_DIR="../../data/test_data_sample"
# 전체 데이터로 실행을 원할 시
# INPUT_DIR="../../data/test_data"
INPUT_DIR="../../data/test_data_fin_korea"

CHAR_DIR="../../data/source_data/meta_character_2_sample.json"
# 전체 데이터로 실행을 원할 시 
CHAR_DIR="../../data/source_data/meta_character.json"

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

# 고정 질문 유형
# QUESTION_TYPES="cross temporal cultural fact"
QUESTION_TYPES="cross cultural"

# 2. 추가: 다양한 context 조합에 대해 실행
CONTEXT_COMBINATIONS=(
  # "no_context"
  "birth"
  "Nationality"
  "Summary"
  "birth Nationality"
  "Nationality Summary"
  "birth Summary"
  # "birth Nationality Summary"
)

for MODEL in "${MODELS[@]}"; do
  for QTYPE in $QUESTION_TYPES; do
    for CONTEXT_TYPES in "${CONTEXT_COMBINATIONS[@]}"; do
      CTX_CLEAN=$(echo "$CONTEXT_TYPES" | tr ' ' '_')  # 예: "birth Nationality" → "birth_Nationality"
      TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
      MODEL_NAME_CLEAN=$(echo "$MODEL" | tr '/ ' '__')
      LOG_FILE_NAME="${MODEL_NAME_CLEAN}_${CTX_CLEAN}_${QTYPE}.log"

      echo "[${TIMESTAMP}] Running model=${MODEL}, question_type=${QTYPE}, context_types=\"${CONTEXT_TYPES}\"..."

      if python script.py \
           --model_name "${MODEL}" \
           --input_dir_path "${INPUT_DIR}" \
           --meta_char_dir "${CHAR_DIR}" \
           --question_type "${QTYPE}" \
           --device_index "${DEVICE}" \
           --context_types ${CONTEXT_TYPES} \
         2>&1 | tee "${LOG_DIR}/${LOG_FILE_NAME}"; then
        echo "  → 성공: 로그 → ${LOG_DIR}/${LOG_FILE_NAME}"
      else
        echo "  **실패**: model=${MODEL}, question_type=${QTYPE}, context_types=\"${CONTEXT_TYPES}\" 실행 중 에러."
        echo "         로그 → ${LOG_DIR}/${LOG_FILE_NAME}"
      fi

      echo
    done
  done
done

echo "모든 작업 완료."

