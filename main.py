import argparse
import base64
import json
import os
import sys
import time
import traceback
import re
import openai
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

console = Console()


class TestResult:
    def __init__(self, status: str, data: Any = None, error: Optional[str] = None):
        self.status = status
        self.data = data
        self.error = error
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


class OpenRouterTester:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.results = {}
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "2"))
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", "120"))
        env_mm = os.getenv("MULTIMODAL_MODELS", "").strip()
        if env_mm:
            self.multimodal_models = [m.strip().lower() for m in env_mm.split(',') if m.strip()]
        else:
            self.multimodal_models = [
                "gpt-5",
                "gpt-4o", "gpt-4-vision", "claude-3", "gemini",
                "llama-3.2-90b-vision", "qwen", "glm-4"
            ]
        
        os.makedirs("out", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.log_file = f"logs/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(self, module: str, method: str, level: str, message: str, error: Optional[Exception] = None):
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] [{module}::{method}] {message}"
        
        if error:
            log_entry += f"\nError: {str(error)}\n{traceback.format_exc()}"
        
        color_map = {
            "DEBUG": "dim",
            "INFO": "cyan", 
            "WARN": "yellow",
            "ERROR": "red"
        }
        
        console.print(f"[{color_map.get(level, 'white')}]{log_entry}[/]")
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    
    def retry_call(self, func, module: str, method: str, description: str):
        for attempt in range(1, self.max_retries + 1):
            try:
                self.log(module, method, "DEBUG", f"Попытка {attempt}/{self.max_retries}: {description}")
                return func()
            except Exception as e:
                if attempt == self.max_retries:
                    self.log(module, method, "ERROR", f"Все попытки исчерпаны для {description}", e)
                    raise
                self.log(module, method, "WARN", f"Ошибка на попытке {attempt}: {e}")
                time.sleep(self.retry_delay * attempt)
    
    def test_chat(self, model: str) -> TestResult:
        module, method = "Chat", "test_basic"
        
        try:
            self.log(module, method, "INFO", f"Тестирование чата для {model}")
            
            messages = [
                {"role": "system", "content": "Ты помощник. Отвечай кратко и по существу."},
                {"role": "user", "content": "Назови столицу Франции и год основания Парижа. Ответь одним предложением."}
            ]
            
            response = self.retry_call(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7
                ),
                module, method, "chat completion"
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            self.log(module, method, "INFO", 
                f"Успешный ответ: {content[:100]}... | Токены: {usage.total_tokens}")
            
            return TestResult("success", {
                "response": content,
                "usage": usage.model_dump() if hasattr(usage, 'model_dump') else None
            })
            
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка чата для {model}", e)
            return TestResult("failed", error=str(e))
    
    def test_stream(self, model: str) -> TestResult:
        module, method = "Stream", "test_sse"
        
        try:
            self.log(module, method, "INFO", f"Тестирование стриминга для {model}")
            
            messages = [
                {"role": "user", "content": "Напиши хайку про программирование на Go."}
            ]
            
            stream = self.retry_call(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=150
                ),
                module, method, "stream creation"
            )
            
            chunks = []
            chunk_count = 0
            
            for chunk in stream:
                chunk_count += 1
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            
            result = "".join(chunks)
            self.log(module, method, "INFO", 
                f"Стрим завершен: {chunk_count} чанков, {len(result)} символов")
            
            return TestResult("success", {
                "response": result,
                "chunks": chunk_count
            })
            
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка стриминга для {model}", e)
            return TestResult("failed", error=str(e))
    
    def test_vision(self, model: str, image_path: str) -> TestResult:
        module, method = "Vision", "test_image"
        
        try:
            if not Path(image_path).exists():
                self.log(module, method, "WARN", f"Изображение {image_path} не найдено")
                return TestResult("skipped", error="Image not found")
            
            lower_model = model.lower()
            is_multimodal = any(mm in lower_model for mm in self.multimodal_models)
            
            if not is_multimodal:
                self.log(module, method, "INFO", f"Пропуск vision для {model} - не мультимодальная")
                return TestResult("skipped", error="Not a multimodal model")
            
            self.log(module, method, "INFO", f"Тестирование vision для {model}")
            
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Опиши что на изображении. Будь конкретным: цвета, объекты, композиция."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }
            ]
            
            response = self.retry_call(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=200
                ),
                module, method, "vision analysis"
            )
            
            content = response.choices[0].message.content
            self.log(module, method, "INFO", f"Vision анализ: {content[:150]}...")
            
            return TestResult("success", {"response": content})
            
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка vision для {model}", e)
            return TestResult("failed", error=str(e))
    
    def test_json_mode(self, model: str) -> TestResult:
        module, method = "JSON", "test_structured"
        
        try:
            self.log(module, method, "INFO", f"Тестирование JSON mode для {model}")
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты возвращаешь только валидный JSON. Никакого текста вне JSON."
                },
                {
                    "role": "user",
                    "content": "Создай JSON с информацией о погоде: city (строка), temperature_c (число), humidity (число 0-100), conditions (строка). Город - Москва."
                }
            ]
            
            response = self.retry_call(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=150
                ),
                module, method, "json generation"
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed = json.loads(content)
                self.log(module, method, "INFO", f"JSON валидный: {json.dumps(parsed, ensure_ascii=False)}")
                return TestResult("success", {"response": parsed})
            except json.JSONDecodeError as e:
                self.log(module, method, "WARN", f"Невалидный JSON: {content[:100]}")
                return TestResult("failed", {"response": content}, error=str(e))
                
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка JSON mode для {model}", e)
            return TestResult("failed", error=str(e))
    
    def test_harmony_format(self, model: str) -> TestResult:
        module, method = "Harmony", "test_structured"
        
        try:
            if "gpt-oss" not in model.lower():
                self.log(module, method, "INFO", f"Пропуск Harmony для {model} - только для gpt-oss моделей")
                return TestResult("skipped", error="Not a gpt-oss model")
            
            self.log(module, method, "INFO", f"Тестирование Harmony format для {model}")
            
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "object",
                        "properties": {
                            "problem": {"type": "string"},
                            "approach": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "conclusion": {"type": "string"}
                        },
                        "required": ["problem", "approach", "steps", "conclusion"]
                    }
                },
                "required": ["reasoning"]
            }
            
            messages = [
                {
                    "role": "system",
                    "content": "Reasoning: high\nТы решаешь задачи пошагово. Возвращай структурированный JSON."
                },
                {
                    "role": "user",
                    "content": "Реши задачу: У Маши было 15 яблок. Она отдала 7 яблок Пете и 3 яблока Кате. Сколько яблок осталось у Маши?"
                }
            ]
            
            response = self.retry_call(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "reasoning_response",
                            "schema": schema,
                            "strict": True
                        }
                    },
                    max_tokens=500
                ),
                module, method, "harmony structured output"
            )

            content = response.choices[0].message.content if response.choices and response.choices[0].message else None

            if not content or not content.strip():
                self.log(module, method, "WARN", "Пустой контент от модели (empty content) — не удалось распарсить JSON")
                return TestResult("failed", {"response": content}, error="Empty response content")

            def try_parse(text: str) -> Tuple[Optional[dict], Optional[str]]:
                """Попытаться распарсить JSON несколькими стратегиями."""
                try:
                    return json.loads(text), None
                except json.JSONDecodeError as err:
                    match = re.search(r'\{[\s\S]*\}', text)
                    if match:
                        candidate = match.group(0)
                        try:
                            return json.loads(candidate), None
                        except json.JSONDecodeError as err_inner:
                            return None, f"Primary parse error: {err}; Extracted segment parse error: {err_inner}"
                    return None, f"Primary parse error: {err}; no JSON object found"

            parsed, parse_error = try_parse(content)
            if parsed is None:
                self.log(module, method, "WARN", f"Невалидный Harmony JSON. Ошибка: {parse_error}. Фрагмент: {content[:160]}")
                return TestResult("failed", {"response": content}, error=parse_error)

            reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
            missing = []
            if isinstance(reasoning, dict):
                for key in ["problem", "approach", "steps", "conclusion"]:
                    if key not in reasoning:
                        missing.append(key)
            else:
                missing.append("reasoning")

            if missing:
                self.log(module, method, "WARN", f"JSON получен, но отсутствуют поля: {', '.join(missing)}")
                return TestResult("failed", {"response": parsed}, error=f"Missing fields: {', '.join(missing)}")

            steps_count = len(reasoning.get("steps", [])) if isinstance(reasoning.get("steps"), list) else 0
            self.log(module, method, "INFO", f"Harmony ответ валиден: {steps_count} шагов")
            return TestResult("success", {"response": parsed})

        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка Harmony для {model}", e)
            return TestResult("failed", error=str(e))
    
    def test_tool_calling(self, model: str) -> TestResult:
        module, method = "Tools", "test_functions"
        
        try:
            lowered = model.lower()

            whitelist_raw = os.getenv("TOOL_MODELS", "").strip()
            if whitelist_raw:
                whitelist = [w.strip().lower() for w in whitelist_raw.split(',') if w.strip()]
                if not any(w in lowered for w in whitelist):
                    self.log(module, method, "INFO", f"Пропуск tool calling для {model} (не в TOOL_MODELS)")
                    return TestResult("skipped", error="Model not in TOOL_MODELS whitelist")
            else:
                heuristic_keys = ["gpt", "claude", "llama", "qwen", "deepseek", "mistral", "command", "sonar", "o3", "reasoning"]
                if not any(k in lowered for k in heuristic_keys):
                    self.log(module, method, "INFO", f"Пропуск tool calling для {model} (не проходит эвристику)")
                    return TestResult("skipped", error="Heuristic skip: likely no tool support")

            self.log(module, method, "INFO", f"Тестирование tool calling для {model}")
            
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Выполнить математическое вычисление",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Математическое выражение для вычисления"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                }
            ]
            
            messages = [
                {"role": "user", "content": "Какая погода в Париже? И сколько будет 25 * 4?"}
            ]
            
            try:
                response = self.retry_call(
                    lambda: self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=200
                    ),
                    module, method, "tool calling"
                )
            except Exception as call_err:
                err_text = str(call_err)
                status_code = getattr(call_err, 'status_code', None)
                if (status_code == 404) or '404' in err_text or 'NotFound' in err_text:
                    self.log(module, method, "WARN", f"Модель не поддерживает tool calling (404). Пропуск.")
                    return TestResult("skipped", error="Tool calling unsupported (404)")
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
                    self.log(module, method, "INFO", f"Tool call: {func_name}({func_args})")
                
                return TestResult("success", {"tool_calls": results})
            else:
                self.log(module, method, "WARN", "Модель не вернула tool calls")
                return TestResult("failed", error="No tool calls returned")
                
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка tool calling для {model}", e)
            return TestResult("failed", error=str(e))

    def test_image_generation(self, model: str, prompt: Optional[str] = None) -> TestResult:
        """Тест генерации изображений для моделей, поддерживающих image generation API.

        По умолчанию пропускаем модели, которые не содержат в названии ключевых слов
        ('image', 'dall', 'gpt-image', 'flux', 'sd', 'stable') чтобы не слать неподдерживаемые запросы.
        """
        module, method = "ImageGen", "test_image_generation"
        try:
            lowered = model.lower()
            if not any(k in lowered for k in ["image", "dall", "gpt-image", "flux", "sd", "stable"]):
                self.log(module, method, "INFO", f"Пропуск image generation для {model} - вероятно не генерирует изображения")
                return TestResult("skipped", error="Model not recognized as image generator")

            prompt = prompt or "A minimalist flat illustration of a friendly cyberpunk cat coding at a holographic laptop, vibrant neon palette"
            self.log(module, method, "INFO", f"Тестирование image generation для {model}")

            start = time.time()
            image_b64 = None
            generation_mode = "images.generate"
            try:
                resp = self.client.images.generate(model=model, prompt=prompt, size="512x512")
                image_b64 = resp.data[0].b64_json if resp and resp.data else None
            except Exception as direct_err:
                self.log(module, method, "WARN", f"Прямой images API не сработал ({direct_err}); fallback через chat попыткой получить data URI")
                generation_mode = "chat-fallback"
                try:
                    chat_resp = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": f"Generate an image for this prompt and return ONLY a JSON object: {{\"image_base64\": <base64 PNG>}}. Prompt: {prompt}"}],
                        max_tokens=1200,
                        temperature=0.7
                    )
                    content = chat_resp.choices[0].message.content
                    try:
                        data = json.loads(content)
                        image_b64 = data.get("image_base64") if isinstance(data, dict) else None
                    except Exception:
                        match = re.search(r"data:image/(?:png|jpeg);base64,([A-Za-z0-9+/=]+)", content or "")
                        if match:
                            image_b64 = match.group(1)
                except Exception as fallback_err:
                    return TestResult("failed", error=f"Both image API and chat fallback failed: {fallback_err}")

            if not image_b64:
                return TestResult("failed", error="No base64 image data returned")

            out_dir = Path("out")
            out_dir.mkdir(exist_ok=True)
            file_path = out_dir / f"generated_{int(time.time())}.png"
            try:
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(image_b64))
            except Exception as write_err:
                return TestResult("failed", error=f"Failed to write image file: {write_err}")

            elapsed = time.time() - start
            self.log(module, method, "INFO", f"Изображение сгенерировано ({generation_mode}) за {elapsed:.2f}с -> {file_path}")
            return TestResult("success", {"file": str(file_path), "mode": generation_mode, "elapsed_seconds": elapsed})
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка image generation для {model}", e)
            return TestResult("failed", error=str(e))
    
    def test_batch(self, model: str, prompts: Optional[List[str]] = None) -> TestResult:
        module, method = "Batch", "test_multiple"
        
        try:
            if not prompts:
                prompts = [
                    "Что такое рекурсия в программировании?",
                    "Объясни принцип DRY одним предложением",
                    "Что такое SOLID принципы?"
                ]
            
            self.log(module, method, "INFO", f"Тестирование batch для {model} с {len(prompts)} запросами")
            
            results = []
            start_time = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Обработка {len(prompts)} запросов...", total=len(prompts))
                
                for i, prompt in enumerate(prompts):
                    try:
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=100
                        )
                        content = response.choices[0].message.content
                        results.append({
                            "prompt": prompt[:50],
                            "response": content[:100],
                            "status": "success",
                            "tokens": response.usage.total_tokens
                        })
                        self.log(module, method, "DEBUG", f"Batch {i+1}/{len(prompts)}: OK")
                    except Exception as e:
                        results.append({
                            "prompt": prompt[:50],
                            "response": None,
                            "status": "failed",
                            "error": str(e)[:100]
                        })
                        self.log(module, method, "WARN", f"Batch {i+1}/{len(prompts)}: FAIL - {e}")
                    
                    progress.advance(task)
            
            elapsed = time.time() - start_time
            success_count = sum(1 for r in results if r["status"] == "success")
            
            self.log(module, method, "INFO", 
                f"Batch завершен: {success_count}/{len(prompts)} успешно за {elapsed:.2f}с")
            
            return TestResult("success" if success_count > 0 else "failed", {
                "total": len(prompts),
                "success": success_count,
                "failed": len(prompts) - success_count,
                "elapsed_seconds": elapsed,
                "results": results
            })
            
        except Exception as e:
            self.log(module, method, "ERROR", f"Ошибка batch для {model}", e)
            return TestResult("failed", error=str(e))
    
    def run_full_test_suite(self, models: List[str], categories: List[str]):
        module, method = "Suite", "run_all"
        
        self.log(module, method, "INFO", f"Запуск полного тестирования: {len(models)} моделей, {len(categories)} категорий")
        
        vision_image = os.getenv("VISION_IMAGE", "payloads/cat.jpeg")
        
        table = Table(title="🚀 OpenRouter API Test Results", box=box.ROUNDED)
        table.add_column("Модель", style="cyan", width=40)
        
        for category in categories:
            table.add_column(category.title(), justify="center", width=12)
        
        for model in models:
            self.results[model] = {}
            row = [model]
            
            console.rule(f"[bold cyan]Тестирование: {model}[/]")
            
            for category in categories:
                if category == "chat":
                    result = self.test_chat(model)
                elif category == "stream":
                    result = self.test_stream(model)
                elif category == "vision":
                    result = self.test_vision(model, vision_image)
                elif category == "json":
                    result = self.test_json_mode(model)
                elif category == "harmony":
                    result = self.test_harmony_format(model)
                elif category == "tools":
                    result = self.test_tool_calling(model)
                elif category == "batch":
                    result = self.test_batch(model)
                elif category == "imagegen":
                    result = self.test_image_generation(model)
                else:
                    result = TestResult("skipped", error=f"Unknown category: {category}")
                
                self.results[model][category] = result.to_dict()
                
                status_icon = {
                    "success": "✅",
                    "failed": "❌",
                    "skipped": "⏭️"
                }.get(result.status, "❓")
                
                row.append(status_icon)
            
            table.add_row(*row)
        
        console.print("\n")
        console.print(table)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"out/test_report_{timestamp}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        self.log(module, method, "INFO", f"Отчет сохранен: {report_file}")
        
        total_tests = len(models) * len(categories)
        success_tests = sum(
            1 for m in self.results.values() 
            for c in m.values() 
            if c["status"] == "success"
        )
        
        console.print(Panel.fit(
            f"[green bold]Тестирование завершено![/]\n\n"
            f"📊 Статистика:\n"
            f"• Моделей протестировано: {len(models)}\n"
            f"• Категорий тестов: {len(categories)}\n"
            f"• Всего тестов: {total_tests}\n"
            f"• Успешных: {success_tests}\n"
            f"• Процент успеха: {(success_tests/total_tests*100):.1f}%\n\n"
            f"📁 Файлы:\n"
            f"• Отчет: {report_file}\n"
            f"• Логи: {self.log_file}",
            title="✨ Итоги тестирования",
            border_style="green"
        ))


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="OpenRouter API Test Suite")
    parser.add_argument("--models", nargs="+", help="Модели для тестирования")
    parser.add_argument("--categories", nargs="+", help="Категории тестов")
    parser.add_argument("--api-key", help="API ключ OpenRouter")
    parser.add_argument("--base-url", help="Base URL для API")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("ROUTER_API_KEY")
    base_url = args.base_url or os.getenv("ROUTER_BASE_URL", "http://localhost:8080/v1")
    
    if not api_key:
        console.print("[red]❌ Ошибка: Не указан API ключ. Установите ROUTER_API_KEY в .env[/]")
        sys.exit(1)
    
    models = args.models or os.getenv("TEST_MODELS", "").split(",")
    if not models or models == ['']:
        models = ["openai/gpt-4o", "openai/gpt-oss-120b", "google/gemma-3n-e4b-it:free"]
    
    categories = args.categories or os.getenv("TEST_CATEGORIES", "").split(",")
    if not categories or categories == ['']:
        categories = ["chat", "stream", "vision", "json", "harmony", "tools", "batch"]
    
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    console.print(Panel.fit(
        f"[bold cyan]OpenRouter API Test Suite[/]\n\n"
        f"🔗 Endpoint: {base_url}\n"
        f"🤖 Модели: {', '.join(models)}\n"
        f"🧪 Тесты: {', '.join(categories)}",
        title="⚙️ Конфигурация",
        border_style="cyan"
    ))
    
    tester = OpenRouterTester(api_key, base_url)
    
    try:
        tester.run_full_test_suite(models, categories)
    except KeyboardInterrupt:
        tester.log("Main", "main", "WARN", "Тестирование прервано пользователем")
        console.print("\n[yellow]⚠️ Тестирование прервано[/]")
        sys.exit(130)
    except Exception as e:
        tester.log("Main", "main", "ERROR", "Критическая ошибка", e)
        console.print(f"\n[red]💥 Критическая ошибка: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()