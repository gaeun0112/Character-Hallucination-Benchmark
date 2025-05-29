#!/usr/bin/env sh
set -e

LOG_DIR="./log"
mkdir -p "$LOG_DIR"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DEVICE=1
# POSIX 호환: 공백으로 구분된 단일 문자열
# CONTEXT_TYPES="birth Nationality Summary"
CONTEXT_TYPES="birth"

for TYPE in temporal cross cultural fact; do
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${TIMESTAMP}] Running question_type=${TYPE}..."

  # --context_types 바로 뒤에 $CONTEXT_TYPES 를 unquoted 로 써서
  # 쉘이 공백 기준으로 분리해 줍니다.
  if python evaluate_all.py \
       --model_name "$MODEL" \
       --question_type "$TYPE" \
       --device_index "$DEVICE" \
       --context_types $CONTEXT_TYPES \
     2>&1 | tee "$LOG_DIR/${TYPE}.log"; then
    echo "  → 성공: 로그 → $LOG_DIR/${TYPE}.log"
  else
    echo "  **실패**: $TYPE 타입 실행 중 에러. 로그 → $LOG_DIR/${MODEL}_${CONTEXT_TYPES}_${TYPE}.log"
  fi

  echo
done

echo "모든 작업 완료."

