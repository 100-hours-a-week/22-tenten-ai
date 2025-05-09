import requests
import os, time
from dotenv import load_dotenv
import logging

class KoalphaLoader:
    def __init__(self, mode="colab"):
        self.mode = mode
        self.model_path = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
        # colab/ngrok API 요청에 필요한 헤더 및 데이터
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-key"  
        }
        self.data = {
            "model": self.model_path ,
            "messages": [],
            "temperature": 0.7,
            "max_tokens": 256
        }
        # GCP(vllm) 모드일 때만 vllm 엔진 초기화
        if self.mode == "gcp":
            from vllm import LLM, SamplingParams

            self.model_vllm = LLM(
                model=self.model_path ,
                dtype="auto", # 또는 torch.bfloat16, torch.float16 등. torch.bfloat16은 최신 GPU(Ampere 이상)에서만 지원됨.
                trust_remote_code=True,
                tensor_parallel_size=1
            )

            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
                stop=["\n\n", "</s>"] 
            )

    def get_response(self, messages):
        if self.mode == "colab":
            # Ngrok(Colab) API로 요청
            load_dotenv(override=True)
            base_url = os.getenv('MODEL_NGROK_URL')
            self.data["messages"] = messages
            url = f"{base_url}/v1/chat/completions"
            start_time = time.time()
            response = requests.post(url, headers=self.headers, json=self.data)
            end_time = time.time()
            print(f"response time : {(end_time - start_time):.3f}")
            if response.status_code != 200:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = response.text
                return {
                    "status_code": response.status_code,
                    "url": response.url,
                    "headers": dict(response.headers),
                    "error": error_body
                }
            body = response.json()
            return {
                "status_code": response.status_code,
                "url": response.url,
                "content": body["choices"][0]["message"]["content"]
            }
        elif self.mode == "gcp":
            '''
            vllm 엔진을 통한 직접 추론
            '''

            # Chat Prompt 형식에서 -> Text Prompt 형식으로 변경
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # messages -> 텍스트로 변환
            text_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # 마지막에 assistant 응답 위치를 표시해줌
                tokenize=False               # string 그대로 받기
            )

            start_time = time.time()

            try:
                outputs = self.model_vllm.generate(text_prompt, self.sampling_params)
                content = outputs[0].outputs[0].text
                
            except Exception as e:
                print(f"ChatCompletion error: {e}")
                return {
                    "status_code": 500,
                    "url": "local_vllm",
                    "error": str(e)
                }
            
            print(f"response time : {time.time() - start_time:.3f} sec")

            return {
                "status_code": 200,
                "url": "local_vllm",
                "content": content
            }

'''
KoalphaLoader 클래스 사용 방법 예시
'''
'''
messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "안녕! 날씨 어때?"}
            ]
koalpha=KoalphaLoader()
print(koalpha.get_response(messages))
'''