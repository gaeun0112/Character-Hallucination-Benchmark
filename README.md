## fewshot&cot 실험 적용 방법

`./code/evaluation/fewshot_cot_test_gpt.sh`와 `./code/evaluation/fewshot_cot_test_vllm.sh`에서 `PROMPT_TEMPLATE_PATHS` 리스트 수정하며 적용 가능.
원하는 프롬프트를 `mc_eval_template_{mode}.txt`로 `./prompt/` 폴더 아래에 저장하면 되며, ouput folder는 `./data/prediction_data/` 아래에 `{모델이름}_{mode}` 이라는 폴더명으로 자동으로 저장된다.