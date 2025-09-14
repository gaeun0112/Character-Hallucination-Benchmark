### 환경 세팅
---
```
setup_conda_env.sh
```
```
pip install -r requirements.txt
```


### config.yaml  파일 설정
---
```
openai_key: {YOUR_OPENAI_KEY}
huggingface_token: {YOUR_HUGGINGFACE_TOKEN}
```

### 실험 list
- 모든 실험들은 실행하기 전에 `code/evaluation` 폴더로 이동 후 실행.
- 코드를 실행하기 전에, 각 `.sh` 파일의 세팅 자세히 확인 후 실행하는 것을 권장.
---
##### 1. Context 요소 유무에 따른 결과
```
# gpt 실험들
bash context_test_gpt.sh
# open source model들 실험
bash context_test_vllm.sh
```
##### 2. Few-shot & CoT 실험 결과 
```
# gpt 실험들
bash fewshot_cot_test_gpt.sh
# open source model들 실험
bash fewshot_cot_test_vllm.sh
```
##### 3. Reasoning model들 실험 결과 
```
# gpt 실험들
bash reasoning_test_gpt.sh
# open source model들 실험
bash _reasoning_test_vllm.sh
```
##### 4. Local language 실험 결과 
`⚠️local language 실험의 경우, 데이터 번역 코드 수행 후 수행해줘야 한다.`
```
# gpt 실험들
bash local_lang_test_gpt.sh
# open source model들 실험
bash local_lang_test_vllm.sh
```

---
`실험 list`의 모든 코드들은 `run_all_evaluation_gpt.sh` 파일과 `run_all_evaluation_vllm.sh` 파일을 통해 한 번에 실행할 수 있다.