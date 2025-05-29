#!/usr/bin/env sh
set -e

LOG_DIR="./log"
mkdir -p "$LOG_DIR"

# 실행할 모델 목록
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "gpt-4o"
  "gpt-3.5-turbo-0125"
  "mistralai/Mistral-Nemo-Instruct-2407"
  "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
)

DEVICE=1

# 사용할 컨텍스트 옵션
CONTEXT_TYPES=("birth" "Nationality" "Summary")

# 컨텍스트 타입의 모든 비어 있지 않은 조합 생성
generate_combinations() {
  local -n arr=$1
  local n=${#arr[@]}
  local combos=()

  for ((mask=1; mask < (1<<n); mask++)); do
    combo=""
    for ((i=0; i<n; i++)); do
      if (( mask & (1<<i) )); then
        combo="${combo:+$combo }${arr[i]}"
      fi
    done
    combos+=("$combo")
  done

  printf '%s\n' "${combos[@]}"
}

# 질문 유형 리스트
QUESTION_TYPES="cross temporal cultural fact"

for MODEL in "${MODELS[@]}"; do
  for QTYPE in $QUESTION_TYPES; do
      TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
      LOG_FILE_NAME=$(echo "${MODEL}_${CONTEXT_TYPES}_${QTYPE}" | tr '/ ' '__')
      echo "[${TIMESTAMP}] Running model=${MODEL}, question_type=${QTYPE}, context_types=\"${CONTEXT_TYPES}\"..."

      if python evaluate_all.py \
           --model_name "${MODEL}" \
           --question_type "${QTYPE}" \
           --device_index "${DEVICE}" \
           --context_types ${CONTEXT_TYPES} \
         2>&1 | tee "${LOG_DIR}/${LOG_FILE_NAME}.log"; then
        echo "  → 성공: 로그 → ${LOG_DIR}/${LOG_FILE_NAME}.log"
      else
        echo "  **실패**: model=${MODEL}, context_types=\"${CONTEXT_TYPES}\", question_type=${QTYPE} 실행 중 에러."
        echo "         로그 → ${LOG_DIR}/${LOG_FILE_NAME}.log"
      fi

      echo
    done
  done
done

echo "모든 작업 완료."