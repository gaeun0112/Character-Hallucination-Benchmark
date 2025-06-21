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

MODELS=(
  "gpt-4o"
  "gpt-3.5-turbo-0125"
)

DEVICE="0,1"
# QUESTION_TYPES="cross temporal cultural fact"
QUESTION_TYPES="temporal fact"
CONTEXT_COMBINATIONS=(
  "no_context"
  "birth"
  "Nationality"
  "Summary"
  "birth Nationality"
  "Nationality Summary"
  "birth Summary"
  "birth Nationality Summary"
)

for MODEL in "${MODELS[@]}"; do
  echo "=== Starting MODEL=$MODEL ==="

  # question_type별로 병렬 실행
  for QTYPE in $QUESTION_TYPES; do
    (
      for CONTEXT_TYPES in "${CONTEXT_COMBINATIONS[@]}"; do
        CTX_CLEAN=$(echo "$CONTEXT_TYPES" | tr ' ' '_')
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
    ) &   # 여기서 백그라운드로 실행
  done

  # 같은 MODEL 내의 모든 QTYPE 작업이 끝날 때까지 대기
  wait

  echo "=== Finished MODEL=$MODEL ==="
  echo
done

echo "모든 작업 완료."