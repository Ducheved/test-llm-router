"""
Image Generation Test Module
Тестирование возможностей генерации изображений
"""
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core import APITestModule, TestResult, TestStatus, ModelCapabilityDetector


class ImageGenerationTestModule(APITestModule):
    """Тестирование генерации изображений через различные API"""
    
    def __init__(self, config, http_client):
        super().__init__(config, http_client)
        self.detector = ModelCapabilityDetector(config, http_client)
    
    def get_test_methods(self) -> List[str]:
        return [
            "direct_image_generation",
            "chat_based_generation", 
            "different_sizes",
            "style_variations",
            "batch_generation"
        ]
    
    def run_test(self, test_name: str, model: str, **kwargs) -> TestResult:
        """Запустить конкретный тест"""
        if not self.detector.supports_image_generation(model):
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Model does not support image generation"
            )
        
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
            self.log("ERROR", f"Image generation test {test_name} failed for {model}", e)
            return TestResult(
                status=TestStatus.ERROR, 
                error=str(e)
            )
    
    def _test_direct_image_generation(self, model: str, **kwargs) -> TestResult:
        """Тест прямой генерации изображений через images API"""
        self.log("INFO", f"Тестирование direct_image_generation для {model}")
        
        prompt = kwargs.get("prompt", "A minimalist flat illustration of a friendly cyberpunk cat coding at a holographic laptop, vibrant neon palette")
        image_data = {
            "model": model,
            "prompt": prompt,
            "size": "512x512",
            "n": 1
        }
        
        def make_image_request():
            return self.make_request("POST", "images/generations", json=image_data)
        
        try:
            response = self.retry_operation(make_image_request, "direct image generation")
        except Exception as e:
            return self._fallback_to_chat_generation(model, prompt)
        
        if response.status_code == 404:
            self.log("WARN", "Images API не найден, переходим к chat fallback")
            return self._fallback_to_chat_generation(model, prompt)
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Image generation failed: {response.status_code}",
                data={"response": response.text[:500]}
            )
        
        result = response.json()
        
        if not result.get("data") or not result["data"]:
            return TestResult(
                status=TestStatus.FAILED,
                error="No image data returned",
                data={"response": result}
            )
        
        image_info = result["data"][0]
        image_b64 = image_info.get("b64_json")
        
        if not image_b64:
            return TestResult(
                status=TestStatus.FAILED,
                error="No base64 image data",
                data={"response": result}
            )
        image_path = self._save_generated_image(image_b64, f"direct_{int(time.time())}")
        
        self.log("SUCCESS", f"Изображение сгенерировано через direct API -> {image_path}")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "prompt": prompt,
                "image_path": str(image_path),
                "method": "direct_api",
                "model_used": result.get("model", model)
            }
        )
    
    def _test_chat_based_generation(self, model: str, **kwargs) -> TestResult:
        """Тест генерации изображений через chat API"""
        self.log("INFO", f"Тестирование chat_based_generation для {model}")
        
        prompt = kwargs.get("prompt", "A serene landscape with mountains and a lake, digital art style")
        
        messages = [
            {
                "role": "user",
                "content": f"Generate an image: {prompt}. Return ONLY a JSON object with 'image_base64' field containing the base64 PNG data."
            }
        ]
        
        chat_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        def make_chat_request():
            return self.make_request("POST", "chat/completions", json=chat_data)
        
        response = self.retry_operation(make_chat_request, "chat-based image generation")
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Chat generation failed: {response.status_code}",
                data={"response": response.text[:500]}
            )
        
        result = response.json()
        
        if not result.get("choices"):
            return TestResult(
                status=TestStatus.FAILED,
                error="No choices returned",
                data={"response": result}
            )
        
        content = result["choices"][0].get("message", {}).get("content", "")
        image_b64 = self._extract_image_data(content)
        
        if not image_b64:
            return TestResult(
                status=TestStatus.FAILED,
                error="No valid image data found in response",
                data={"response": content[:500]}
            )
        image_path = self._save_generated_image(image_b64, f"chat_{int(time.time())}")
        
        self.log("SUCCESS", f"Изображение сгенерировано через chat API -> {image_path}")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "prompt": prompt,
                "image_path": str(image_path),
                "method": "chat_api",
                "response_excerpt": content[:200]
            }
        )
    
    def _test_different_sizes(self, model: str, **kwargs) -> TestResult:
        """Тест генерации изображений разных размеров"""
        self.log("INFO", f"Тестирование different_sizes для {model}")
        
        sizes = ["256x256", "512x512", "1024x1024"]
        prompt = "Simple geometric pattern, abstract art"
        
        results = []
        
        for size in sizes:
            try:
                image_data = {
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "n": 1
                }
                
                def make_size_request():
                    return self.make_request("POST", "images/generations", json=image_data)
                
                response = self.retry_operation(make_size_request, f"image generation {size}")
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("data") and result["data"]:
                        image_b64 = result["data"][0].get("b64_json")
                        if image_b64:
                            image_path = self._save_generated_image(image_b64, f"size_{size}_{int(time.time())}")
                            results.append({
                                "size": size,
                                "status": "success",
                                "path": str(image_path)
                            })
                        else:
                            results.append({"size": size, "status": "no_data"})
                    else:
                        results.append({"size": size, "status": "no_data"})
                else:
                    results.append({
                        "size": size, 
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    })
            
            except Exception as e:
                results.append({
                    "size": size,
                    "status": "error", 
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        self.log("SUCCESS" if success_count > 0 else "WARN", 
               f"Size variations: {success_count}/{len(sizes)} successful")
        
        return TestResult(
            status=TestStatus.SUCCESS if success_count > 0 else TestStatus.FAILED,
            data={
                "prompt": prompt,
                "sizes_tested": sizes,
                "results": results,
                "success_count": success_count
            }
        )
    
    def _fallback_to_chat_generation(self, model: str, prompt: str) -> TestResult:
        """Fallback генерация через chat API"""
        self.log("INFO", f"Fallback to chat generation для {model}")
        
        messages = [
            {
                "role": "user",
                "content": f"Generate this image: {prompt}. Return the result as base64 PNG data."
            }
        ]
        
        chat_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.7
        }
        
        def make_fallback_request():
            return self.make_request("POST", "chat/completions", json=chat_data)
        
        response = self.retry_operation(make_fallback_request, "chat fallback generation")
        
        if response.status_code != 200:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Chat fallback failed: {response.status_code}"
            )
        
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        image_b64 = self._extract_image_data(content)
        
        if image_b64:
            image_path = self._save_generated_image(image_b64, f"fallback_{int(time.time())}")
            return TestResult(
                status=TestStatus.SUCCESS,
                data={
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "method": "chat_fallback"
                }
            )
        else:
            return TestResult(
                status=TestStatus.FAILED,
                error="No image data in chat fallback",
                data={"response": content[:300]}
            )
    
    def _extract_image_data(self, content: str) -> Optional[str]:
        """Извлечь base64 данные изображения из текста"""
        import re
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "image_base64" in data:
                return data["image_base64"]
        except:
            pass
        match = re.search(r"data:image/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)", content)
        if match:
            return match.group(1)
        match = re.search(r"([A-Za-z0-9+/]{100,}={0,2})", content)
        if match:
            return match.group(1)
        
        return None
    
    def _save_generated_image(self, image_b64: str, filename_prefix: str) -> Path:
        """Сохранить сгенерированное изображение"""
        try:
            image_data = base64.b64decode(image_b64)
            
            output_path = self.config.output_dir / f"{filename_prefix}.png"
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            return output_path
        
        except Exception as e:
            self.log("ERROR", f"Failed to save image: {e}")
            return self.config.output_dir / f"{filename_prefix}_failed.png"

