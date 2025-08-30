"""
Chat Completion    def __init__(self, config, http_client):
        super().__init__(config, http_client)
        if OpenAI:
            self.openai_client = OpenAI(
                api_key=config.api_key, 
                base_url=config.base_url
            )st Module
Тестирование основного API для чат-завершений
"""
import json
import re
import base64
from typing import Dict, List, Optional, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..core import APITestModule, TestResult, TestStatus


class ChatCompletionsTestModule(APITestModule):
    """Тестирование /api/v1/chat/completions endpoint"""
    
    def __init__(self, config, http_client):
        super().__init__(config, http_client)
        if OpenAI:
            self.openai_client = OpenAI(
                api_key=config.api_key, 
                base_url=config.base_url
            )
        else:
            self.openai_client = None
    
    def get_test_methods(self) -> List[str]:
        return [
            "basic_chat",
            "streaming_chat", 
            "system_message",
            "json_mode",
            "vision_analysis",
            "tool_calling",
            "multi_turn_conversation",
            "assistant_prefill",
            "temperature_variations"
        ]
    
    def run_test(self, test_name: str, model: str, **kwargs) -> TestResult:
        """Запустить конкретный тест"""
        method_name = f"_test_{test_name}"
        if not hasattr(self, method_name):
            return TestResult(
                status=TestStatus.ERROR,
                error=f"Test method {test_name} not found"
            )
        
        try:
            test_method = getattr(self, method_name)
            result, duration = self.time_operation(lambda: test_method(model, **kwargs))
            result.duration = duration
            return result
        except Exception as e:
            self.log("ERROR", f"Test {test_name} failed for {model}", e)
            return TestResult(
                status=TestStatus.ERROR, 
                error=str(e),
                metadata={"traceback": str(e)}
            )
    
    def _test_basic_chat(self, model: str, **kwargs) -> TestResult:
        """Базовый тест чат-завершения"""
        self.log("INFO", f"Тестирование basic_chat для {model}")
        if self.openai_client:
            messages = [
                {"role": "system", "content": "Ты помощник. Отвечай кратко и по существу."},
                {"role": "user", "content": "Назови столицу Франции и год основания Парижа. Ответь одним предложением."}
            ]
            
            def make_request():
                return self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7
                )
            
            try:
                response = self.retry_operation(make_request, "basic chat completion")
                
                content = response.choices[0].message.content
                usage = response.usage
                
                self.log("SUCCESS", f"Получен ответ: {content[:100]}...")
                
                return TestResult(
                    status=TestStatus.SUCCESS,
                    data={
                        "response": content,
                        "usage": usage.model_dump() if hasattr(usage, 'model_dump') else usage.__dict__,
                        "model": response.model,
                        "finish_reason": response.choices[0].finish_reason
                    },
                    metadata={"response_length": len(content)}
                )
            except Exception as e:
                return self._fallback_to_http_chat(model, messages)
        messages = [
            {"role": "system", "content": "Ты помощник. Отвечай кратко и по существу."},
            {"role": "user", "content": "Назови столицу Франции и год основания Парижа. Ответь одним предложением."}
        ]
        return self._fallback_to_http_chat(model, messages)
        
    def _fallback_to_http_chat(self, model: str, messages: list) -> TestResult:
        """HTTP fallback для chat запросов"""
        self.log("INFO", "Переход к HTTP запросу")
        chat_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        def make_request():
            return self.make_request("POST", "chat/completions", json=chat_data)
        
        response = self.retry_operation(make_request, "HTTP chat completion")
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"HTTP {response.status_code}: {response.text[:200]}"
            )
        
        result = response.json()
        
        if not result.get("choices"):
            return TestResult(
                status=TestStatus.FAILED,
                error="No choices in response",
                data={"response": result}
            )
        
        content = result["choices"][0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        
        self.log("SUCCESS", f"HTTP ответ: {content[:100]}...")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "response": content,
                "usage": usage,
                "model": result.get("model", model),
                "finish_reason": result["choices"][0].get("finish_reason")
            },
            metadata={"response_length": len(content), "method": "http"}
        )
    
    def _test_streaming_chat(self, model: str, **kwargs) -> TestResult:
        """Тест стриминга чат-завершения"""
        self.log("INFO", f"Тестирование streaming_chat для {model}")
        
        messages = [
            {"role": "user", "content": "Напиши хайку про программирование на Python."}
        ]
        if self.openai_client:
            def make_stream_request():
                return self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=150
                )
            
            try:
                stream = self.retry_operation(make_stream_request, "streaming chat")
                
                chunks = []
                chunk_count = 0
                
                for chunk in stream:
                    chunk_count += 1
                    if chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                
                result_text = "".join(chunks)
                
                self.log("SUCCESS", f"Стрим завершен: {chunk_count} чанков, {len(result_text)} символов")
                
                return TestResult(
                    status=TestStatus.SUCCESS,
                    data={
                        "response": result_text,
                        "chunks_received": chunk_count,
                        "total_length": len(result_text),
                        "method": "openai_client"
                    }
                )
                
            except Exception as e:
                self.log("WARN", f"OpenAI streaming failed: {e}")
                return self._fallback_to_regular_chat(model, messages)
        else:
            return self._fallback_to_regular_chat(model, messages)
    
    def _test_json_mode(self, model: str, **kwargs) -> TestResult:
        """Тест JSON режима"""
        self.log("INFO", f"Тестирование json_mode для {model}")
        
        messages = [
            {
                "role": "system",
                "content": "Ты должен отвечать ТОЛЬКО валидным JSON. Никакого markdown, никакого текста до или после JSON."
            },
            {
                "role": "user", 
                "content": "Создай JSON с информацией о погоде в Москве. Поля: city, temperature_c, humidity, conditions. Ответь ТОЛЬКО JSON без форматирования."
            }
        ]
        def make_json_request_with_format():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.1
            )
        
        def make_json_request_without_format():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )
        
        try:
            response = self.retry_operation(make_json_request_with_format, "JSON mode completion")
            method = "with_response_format"
        except Exception as e:
            self.log("WARN", f"JSON mode с response_format не работает: {e}")
            try:
                response = self.retry_operation(make_json_request_without_format, "JSON completion fallback")
                method = "without_response_format"
            except Exception as e2:
                return TestResult(
                    status=TestStatus.ERROR,
                    error=f"Both JSON methods failed: {str(e2)}"
                )
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content.strip()
            else:
                return TestResult(
                    status=TestStatus.ERROR,
                    error=f"Invalid response type: {type(response)}, content: {str(response)[:100]}"
                )
        except (AttributeError, IndexError) as e:
            return TestResult(
                status=TestStatus.ERROR,
                error=f"Error accessing response content: {e}, response type: {type(response)}"
            )
        if "```json" in content:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        elif "```" in content:
            content = re.sub(r'```[^`]*```', '', content).strip()
        
        try:
            parsed_json = json.loads(content)
            self.log("SUCCESS", f"JSON валидный: {json.dumps(parsed_json, ensure_ascii=False)}")
            
            return TestResult(
                status=TestStatus.SUCCESS,
                data={
                    "response": parsed_json,
                    "raw_content": response.choices[0].message.content,
                    "cleaned_content": content,
                    "is_valid_json": True,
                    "method": method
                }
            )
        except json.JSONDecodeError as e:
            self.log("WARN", f"Невалидный JSON: {content[:100]}")
            
            return TestResult(
                status=TestStatus.FAILED,
                data={
                    "response": content,
                    "raw_response": response.choices[0].message.content,
                    "is_valid_json": False,
                    "method": method
                },
                error=f"JSON decode error: {str(e)}"
            )
    
    def _test_vision_analysis(self, model: str, **kwargs) -> TestResult:
        """Тест анализа изображений"""
        from ..core import ModelCapabilityDetector
        
        detector = ModelCapabilityDetector(self.config, self.http)
        
        if not detector.is_multimodal(model):
            self.log("INFO", f"Пропуск vision_analysis для {model} - не мультимодальная")
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Not a multimodal model"
            )
        
        image_path = self.config.vision_image_path
        if not (self.config.payloads_dir / "cat.jpeg").exists():
            return TestResult(
                status=TestStatus.SKIPPED,
                error=f"Image file not found: {image_path}"
            )
        
        self.log("INFO", f"Тестирование vision_analysis для {model}")
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Опиши что на изображении. Будь конкретным: цвета, объекты, композиция."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }
        ]
        
        def make_vision_request():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200
            )
        
        response = self.retry_operation(make_vision_request, "vision analysis")
        
        content = response.choices[0].message.content
        
        self.log("SUCCESS", f"Vision анализ: {content[:150]}...")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "response": content,
                "image_used": image_path,
                "response_length": len(content)
            }
        )
    
    def _test_tool_calling(self, model: str, **kwargs) -> TestResult:
        """Тест вызова функций"""
        self.log("INFO", f"Тестирование tool_calling для {model}")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Получить текущую погоду в городе",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Город, например 'Москва'"
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Единицы измерения"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "Какая погода в Париже?"}
        ]
        
        def make_tool_request():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=200
            )
        
        try:
            response = self.retry_operation(make_tool_request, "tool calling")
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                self.log("INFO", f"Модель {model} не поддерживает tool calling (404)")
                return TestResult(
                    status=TestStatus.SKIPPED,
                    error="Tool calling not supported"
                )
            raise
        
        tool_calls = response.choices[0].message.tool_calls
        
        if tool_calls:
            results = []
            for call in tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)
                results.append({
                    "tool": func_name,
                    "arguments": func_args,
                    "id": call.id
                })
                self.log("SUCCESS", f"Tool call: {func_name}({func_args})")
            
            return TestResult(
                status=TestStatus.SUCCESS,
                data={"tool_calls": results}
            )
        else:
            return TestResult(
                status=TestStatus.FAILED,
                error="No tool calls returned"
            )
    
    def _fallback_to_regular_chat(self, model: str, messages: list) -> TestResult:
        """Fallback к обычному chat запросу если streaming не работает"""
        self.log("INFO", "Переход к обычному chat запросу")
        
        chat_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 150
        }
        
        def make_regular_request():
            return self.make_request("POST", "chat/completions", json=chat_data)
        
        try:
            response = self.retry_operation(make_regular_request, "regular chat fallback")
            
            if response.status_code != 200:
                return TestResult(
                    status=TestStatus.FAILED,
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
            
            result = response.json()
            
            if not result.get("choices"):
                return TestResult(
                    status=TestStatus.FAILED,
                    error="No choices in response"
                )
            
            content = result["choices"][0].get("message", {}).get("content", "")
            
            return TestResult(
                status=TestStatus.SUCCESS,
                data={
                    "response": content,
                    "method": "fallback_regular",
                    "usage": result.get("usage")
                }
            )
            
        except Exception as e:
            return TestResult(
                status=TestStatus.ERROR,
                error=f"Fallback failed: {str(e)}"
            )

