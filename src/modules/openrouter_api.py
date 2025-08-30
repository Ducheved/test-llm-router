"""
OpenRouter API Endpoints Test Module
Тестирование всех специфичных endpoints OpenRouter
"""
import json
import time
from typing import Dict, List, Optional, Any

from ..core import APITestModule, TestResult, TestStatus


class OpenRouterAPITestModule(APITestModule):
    """Тестирование OpenRouter-специфичных API endpoints"""
    
    def get_test_methods(self) -> List[str]:
        return [
            "generation_stats",
            "models_list",
            "model_endpoints", 
            "completions_api",
            "model_capabilities",
            "provider_routing",
            "rate_limiting"
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
            self.log("ERROR", f"OpenRouter API test {test_name} failed", e)
            return TestResult(
                status=TestStatus.ERROR, 
                error=str(e)
            )
    
    def _test_generation_stats(self, model: str, **kwargs) -> TestResult:
        """Тест получения статистики генерации через /api/v1/generation"""
        self.log("INFO", f"Тестирование generation_stats для {model}")
        chat_data = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Привет! Как дела?"}
            ],
            "max_tokens": 50
        }
        def make_chat_request():
            return self.make_request("POST", "chat/completions", json=chat_data)
        
        chat_response = self.retry_operation(make_chat_request, "chat completion for generation")
        
        if chat_response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Chat completion failed: {chat_response.status_code}",
                data={"response": chat_response.text[:500]}
            )
        
        chat_result = chat_response.json()
        generation_id = chat_result.get("id")
        
        if not generation_id:
            return TestResult(
                status=TestStatus.FAILED,
                error="No generation ID returned",
                data={"response": chat_result}
            )
        def get_generation_stats():
            return self.make_request("GET", f"generation?id={generation_id}")
        time.sleep(1)
        
        stats_response = self.retry_operation(get_generation_stats, "generation stats")
        
        if stats_response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Generation stats failed: {stats_response.status_code}",
                data={"response": stats_response.text[:500]}
            )
        
        stats_data = stats_response.json()
        
        self.log("SUCCESS", f"Generation stats retrieved for ID: {generation_id}")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "generation_id": generation_id,
                "stats": stats_data,
                "chat_response": chat_result.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]
            }
        )
    
    def _test_models_list(self, model: str = None, **kwargs) -> TestResult:
        """Тест получения списка моделей через /api/v1/models"""
        self.log("INFO", "Тестирование models_list")
        
        def get_models():
            return self.make_request("GET", "models")
        
        response = self.retry_operation(get_models, "models list")
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Models list failed: {response.status_code}",
                data={"response": response.text[:500]}
            )
        
        models_data = response.json()
        models_list = models_data.get("data", [])
        
        if not models_list:
            return TestResult(
                status=TestStatus.FAILED,
                error="No models returned",
                data={"response": models_data}
            )
        model_count = len(models_list)
        multimodal_count = sum(1 for m in models_list 
                             if "image" in m.get("architecture", {}).get("input_modalities", []))
        
        self.log("SUCCESS", f"Retrieved {model_count} models, {multimodal_count} multimodal")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "total_models": model_count,
                "multimodal_models": multimodal_count,
                "sample_models": models_list[:5],
                "models_with_tools": sum(1 for m in models_list 
                                       if "tools" in m.get("supported_parameters", []))
            }
        )
    
    def _test_model_endpoints(self, model: str, **kwargs) -> TestResult:
        """Тест получения endpoints для конкретной модели"""
        if "/" not in model:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Model ID should contain author/model format"
            )
        
        self.log("INFO", f"Тестирование model_endpoints для {model}")
        
        author, slug = model.split("/", 1)
        
        def get_model_endpoints():
            return self.make_request("GET", f"models/{author}/{slug}/endpoints")
        
        response = self.retry_operation(get_model_endpoints, f"model endpoints for {model}")
        
        if response.status_code == 404:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Model not found or endpoints not available"
            )
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Model endpoints failed: {response.status_code}",
                data={"response": response.text[:500]}
            )
        
        endpoints_data = response.json()
        
        model_info = endpoints_data.get("data", {})
        endpoints = model_info.get("endpoints", [])
        
        if not endpoints:
            return TestResult(
                status=TestStatus.FAILED,
                error="No endpoints returned for model"
            )
        providers = [ep.get("provider_name") for ep in endpoints]
        avg_context_length = sum(ep.get("context_length", 0) for ep in endpoints) / len(endpoints)
        
        self.log("SUCCESS", f"Model {model} has {len(endpoints)} endpoints across {len(set(providers))} providers")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "model": model,
                "endpoints_count": len(endpoints),
                "providers": list(set(providers)),
                "avg_context_length": avg_context_length,
                "architecture": model_info.get("architecture"),
                "endpoints_sample": endpoints[0] if endpoints else None
            }
        )
    
    def _test_completions_api(self, model: str, **kwargs) -> TestResult:
        """Тест Completions API (не Chat)"""
        self.log("INFO", f"Тестирование completions_api для {model}")
        
        completion_data = {
            "model": model,
            "prompt": "Продолжи предложение: Программирование на Python это",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        def make_completion_request():
            return self.make_request("POST", "completions", json=completion_data)
        
        response = self.retry_operation(make_completion_request, "completions API")
        
        if response.status_code == 404:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Completions API not supported for this model"
            )
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Completions API failed: {response.status_code}",
                data={"response": response.text[:500]}
            )
        
        result = response.json()
        
        if not result.get("choices"):
            return TestResult(
                status=TestStatus.FAILED,
                error="No choices returned",
                data={"response": result}
            )
        
        completion_text = result["choices"][0].get("text", "")
        
        self.log("SUCCESS", f"Completions API: {completion_text[:100]}...")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "prompt": completion_data["prompt"],
                "completion": completion_text,
                "usage": result.get("usage"),
                "model_used": result.get("model")
            }
        )
    
    def _test_model_capabilities(self, model: str, **kwargs) -> TestResult:
        """Тест определения возможностей модели через API metadata"""
        self.log("INFO", f"Тестирование model_capabilities для {model}")
        
        if "/" not in model:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Model ID should contain author/model format"
            )
        
        author, slug = model.split("/", 1)
        
        def get_model_info():
            return self.make_request("GET", f"models/{author}/{slug}")
        
        response = self.retry_operation(get_model_info, f"model info for {model}")
        
        if response.status_code == 404:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Model info not available"
            )
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Model info failed: {response.status_code}"
            )
        
        model_data = response.json()
        model_info = model_data.get("data", {})
        
        architecture = model_info.get("architecture", {})
        input_modalities = architecture.get("input_modalities", [])
        output_modalities = architecture.get("output_modalities", [])
        
        capabilities = {
            "supports_text": "text" in input_modalities,
            "supports_images": "image" in input_modalities,
            "generates_text": "text" in output_modalities,
            "generates_images": "image" in output_modalities,
            "context_length": model_info.get("context_length"),
            "pricing": model_info.get("pricing")
        }
        
        self.log("SUCCESS", f"Model capabilities: {sum(v for v in capabilities.values() if isinstance(v, bool))} features")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "model": model,
                "capabilities": capabilities,
                "architecture": architecture,
                "full_info": model_info
            }
        )

