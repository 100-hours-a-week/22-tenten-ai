import requests
import os, time
from dotenv import load_dotenv
from utils.logger import log_inference_to_langfuse
from vllm.lora.request import LoRARequest
from openai import OpenAI
from abc import ABC, abstractmethod

class BaseModelLoader(ABC):
    @abstractmethod
    def get_response(self, messages, trace, start_time=None, prompt=None, name="vllm-inference", adapter_type="youtube_summary", endpoint_info=None):
        pass

class ColabModelLoader(BaseModelLoader):
    def __init__(self, model_path, temperature, top_p, max_tokens, stop, headers):
        self.model_path = model_path
        self.headers = headers
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        self.data = {
            "model": self.model_path,
            "messages": [],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop
        }

    def get_response(self, messages, trace, start_time=None, prompt=None, name="colab-inference", adapter_type="youtube_summary", endpoint_info=None):
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


class GCPModelLoader(BaseModelLoader):
    def __init__(self, mode, model_path, temperature, top_p, max_tokens, stop, tensor_parallel_size, max_model_len, gpu_memory_utilization, max_num_seqs, max_num_batched_tokens):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.mode = mode
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

        if self.mode == "gcp-prod":
            load_dotenv(dotenv_path='/secrets/env')
        else:
            load_dotenv(override=True)

        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

        self.model_vllm = LLM(
            model=self.model_path,
            enable_lora=True,
            max_loras=2,
            dtype="half",
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            token=hf_token
        )
        
        self.lora_adapters = {
            "youtube_summary" : LoRARequest(
                "article_summary", 
                1, 
                "KakaoBase/HyperCLOVAX-SEED-article-summary-LoRA-v1.0"
            ),
            "social_bot" : LoRARequest(
                "sns_chat",
                2,
                "KakaoBase/HyperCLOVAX-SEED-SNS-chat-LoRA-v1.0"
            )
        }

        print("🔧 사용 가능한 LoRA 어댑터:")
        for adapter_name, adapter in self.lora_adapters.items():
            print(f"  - {adapter_name}: {adapter.lora_name} (ID: {adapter.lora_int_id})")

    def get_response(self, messages, trace, start_time=None, prompt=None, name="vllm-inference", adapter_type="youtube_summary", endpoint_info=None):
        """
        adapter_type: "youtube_summary" 또는 "social_bot"
        endpoint_info: 로깅을 위한 추가 정보 (예: "sns_chat")
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        start_time = time.time()

        try:
            # 🎯 어댑터 타입에 따라 선택
            selected_lora = self.lora_adapters.get(adapter_type)
            if not selected_lora:
                raise ValueError(f"Unknown adapter type: {adapter_type}")
            
            log_endpoint_info = f" ({endpoint_info})" if endpoint_info else ""
            print(f"사용 중인 LoRA 어댑터: {adapter_type}{log_endpoint_info} ({selected_lora.lora_name})")

            if adapter_type == "youtube_summary":
                print(f"DEBUG: Youtube summary lora_request: {selected_lora.lora_name}, ID: {selected_lora.lora_int_id}")
                outputs = self.model_vllm.generate(
                prompt, 
                self.sampling_params, 
                lora_request=selected_lora
            )
            elif adapter_type == "social_bot":

                print(f"DEBUG: Social bot lora_request: {selected_lora.lora_name}, ID: {selected_lora.lora_int_id}")

                try:
                    outputs = self.model_vllm.generate(
                        prompt, 
                        self.sampling_params, 
                        lora_request=selected_lora,
                    )
                except Exception as gen_e:
                    print(f"ERROR: vllm generate call failed: {gen_e}")
                    raise # 다시 예외를 발생시켜 상위 except 블록에서 처리하도록 함

            # outputs 객체 유효성 검사 및 디버그 출력 추가
            if not outputs or len(outputs) == 0 or not hasattr(outputs[0], 'outputs') or len(outputs[0].outputs) == 0:
                raise ValueError("Model did not generate any output or output structure is invalid.")

            output_token_ids = outputs[0].outputs[0].token_ids
            output_tokens = len(output_token_ids)

            content = outputs[0].outputs[0].text

            tokenization_result = self.tokenizer(prompt)
            input_tokens = len(tokenization_result["input_ids"])

            end_time = time.time()
            inference_time = end_time - start_time

            return {
                "status_code": 200,
                "url": "local_vllm",
                "content": content,
                "adapter_used": adapter_type
            }

        except Exception as e:
            import traceback
            error_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            from datetime import datetime
            gen_end = datetime.now()
            print(f"ChatCompletion error: {e}")
            return {
                "status_code": 500,
                "url": "local_vllm",
                "error": str(e)
            }

        print(f"response time : {inference_time:.3f} sec")

        return {
            "status_code": 200,
            "url": "local_vllm",
            "content": content,
            "adapter_used": adapter_type
        }


class GeminiAPILoader(BaseModelLoader):
    def __init__(self, mode, model_path, temperature, top_p, max_tokens, stop, base_url):
        self.mode = mode
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

        if self.mode == "api-prod":
            load_dotenv(dotenv_path='/secrets/env')
        else:
            load_dotenv(override=True)

        self.client = OpenAI(
            api_key= os.getenv("GEMINI_API_KEY"),
            base_url=base_url
        )

    def get_response(self, messages, trace, start_time=None, prompt=None, name="api-inference", adapter_type="youtube_summary", endpoint_info=None):
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop
            )

            print(f"DEBUG: response: {response}")

            content = response.choices[0].message.content

            print(f"DEBUG: content: {content}")

            input_tokens = output_tokens = None

            end_time = time.time()
            inference_time = end_time - start_time

            return {
                "status_code": 200,
                "url": "local_api",
                "content": content
            }

        except Exception as e:
            import traceback
            error_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            from datetime import datetime
            gen_end = datetime.now()
            print(f"ChatCompletion error: {e}")
            return {
                "status_code": 500,
                "url": "local_api",
                "error": str(e)
            }

        print(f"response time : {inference_time:.3f} sec")

        return {
            "status_code": 200,
            "url": "local_api",
            "content": content
        }


class ModelLoader:
    def __init__(self, mode="colab"):
        self.mode = mode
        self.loader = None
        if mode == "colab":
            self.loader = ColabModelLoader(
                model_path="allganize/Llama-3-Alpha-Ko-8B-Instruct",
                temperature=0.5,
                top_p=0.5,
                max_tokens=256,
                stop=["\n\n", "</s>"],
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer dummy-key"
                }
            )
        elif mode == "gcp-dev" or mode == "gcp-prod":
            self.loader = GCPModelLoader(
                mode=mode,
                model_path="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                temperature=0.5,
                top_p=0.5,
                max_tokens=256,
                stop=["\n\n", "</s>", "\n"],
                tensor_parallel_size=1,
                max_model_len=8192,
                gpu_memory_utilization=0.9,
                max_num_seqs=5,
                max_num_batched_tokens=2048
            )
        elif mode == "api-dev" or mode == "api-prod":
            print("ModelLoader init")
            self.loader = GeminiAPILoader(
                mode=mode,
                model_path="models/gemini-2.0-flash",
                temperature=0.5,
                top_p=0.5,
                max_tokens=256,
                stop=["\n"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_response(self, messages, trace, start_time=None, prompt=None, name="inference", adapter_type="youtube_summary", endpoint_info=None):
        if self.loader:
            return self.loader.get_response(messages, trace, start_time, prompt, name, adapter_type, endpoint_info)
        else:
            raise RuntimeError("Model loader not initialized.")