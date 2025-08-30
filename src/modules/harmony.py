"""
Harmony Format Test Module
Специализированный модуль для тестирования Harmony format (gpt-oss модели)
"""
import json
import re
from typing import Dict, List, Optional, Any, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..core import APITestModule, TestResult, TestStatus, ModelCapabilityDetector


class HarmonyTestModule(APITestModule):
    """Тестирование Harmony format для reasoning моделей"""
    
    def __init__(self, config, http_client):
        super().__init__(config, http_client)
        if OpenAI:
            self.openai_client = OpenAI(
                api_key=config.api_key, 
                base_url=config.base_url
            )
        else:
            self.openai_client = None
        self.detector = ModelCapabilityDetector(config, http_client)
    
    def get_test_methods(self) -> List[str]:
        return [
            "reasoning_structured",
            "high_reasoning",
            "channel_separation",
            "step_by_step_analysis",
            "complex_problem_solving"
        ]
    
    def run_test(self, test_name: str, model: str, **kwargs) -> TestResult:
        """Запустить конкретный тест"""
        if not self.detector.supports_reasoning(model):
            return TestResult(
                status=TestStatus.SKIPPED,
                error="Model does not support reasoning/harmony format"
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
            self.log("ERROR", f"Harmony test {test_name} failed for {model}", e)
            return TestResult(
                status=TestStatus.ERROR, 
                error=str(e)
            )
    
    def _test_reasoning_structured(self, model: str, **kwargs) -> TestResult:
        """Тест структурированного reasoning вывода"""
        self.log("INFO", f"Тестирование reasoning_structured для {model}")
        
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
                "content": "Reasoning: high\\nТы решаешь задачи пошагово. Возвращай структурированный JSON."
            },
            {
                "role": "user",
                "content": "Реши задачу: У Маши было 15 яблок. Она отдала 7 яблок Пете и 3 яблока Кате. Сколько яблок осталось у Маши?"
            }
        ]
        
        def make_reasoning_request():
            return self.openai_client.chat.completions.create(
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
            )
        
        response = self.retry_operation(make_reasoning_request, "harmony structured output")
        
        content = response.choices[0].message.content
        
        if not content or not content.strip():
            return TestResult(
                status=TestStatus.FAILED,
                error="Empty response content",
                data={"response": content}
            )
        
        parsed, parse_error = self._try_parse_json(content)
        if parsed is None:
            return TestResult(
                status=TestStatus.FAILED,
                error=parse_error,
                data={"response": content}
            )
        reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
        missing_fields = []
        
        if isinstance(reasoning, dict):
            for field in ["problem", "approach", "steps", "conclusion"]:
                if field not in reasoning:
                    missing_fields.append(field)
        else:
            missing_fields.append("reasoning")
        
        if missing_fields:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Missing required fields: {', '.join(missing_fields)}",
                data={"response": parsed}
            )
        
        steps_count = len(reasoning.get("steps", [])) if isinstance(reasoning.get("steps"), list) else 0
        
        self.log("SUCCESS", f"Harmony structured response: {steps_count} reasoning steps")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "response": parsed,
                "steps_count": steps_count,
                "has_all_fields": len(missing_fields) == 0
            }
        )
    
    def _test_high_reasoning(self, model: str, **kwargs) -> TestResult:
        """Тест high-level reasoning с анализом каналов"""
        self.log("INFO", f"Тестирование high_reasoning для {model}")
        system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-20
Reasoning: high

Valid channels: analysis, commentary, final. Channel must be included for every message."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": "Объясни почему 2+2=4, используя математические принципы. Покажи пошаговое рассуждение."
            }
        ]
        
        def make_high_reasoning_request():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
        
        response = self.retry_operation(make_high_reasoning_request, "high reasoning")
        
        content = response.choices[0].message.content
        channels_found = self._analyze_harmony_channels(content)
        reasoning_quality = self._analyze_reasoning_quality(content)
        
        self.log("SUCCESS", f"High reasoning completed, channels: {list(channels_found.keys())}")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "response": content,
                "channels_detected": channels_found,
                "reasoning_analysis": reasoning_quality,
                "response_length": len(content)
            }
        )
    
    def _test_channel_separation(self, model: str, **kwargs) -> TestResult:
        """Тест правильного разделения каналов в Harmony format"""
        self.log("INFO", f"Тестирование channel_separation для {model}")
        system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-20
Reasoning: high

Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'."""
        
        developer_prompt = """# Instructions
Analyze the user's request step by step and provide reasoning.
namespace functions {
  // Gets the current weather in the provided location
  type get_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    format?: "celsius" | "fahrenheit", // default: celsius
  }) => any;
}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "developer", "content": developer_prompt},
            {
                "role": "user",
                "content": "Как погода в Москве? Нужно ли брать зонт?"
            }
        ]
        
        def make_channel_request():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=600
            )
        
        response = self.retry_operation(make_channel_request, "channel separation test")
        
        content = response.choices[0].message.content
        channels = self._analyze_harmony_channels(content)
        expected_channels = ["analysis", "commentary"]
        
        found_expected = sum(1 for ch in expected_channels if ch in channels)
        
        self.log("SUCCESS", f"Channel separation test: found {list(channels.keys())}")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "response": content,
                "channels_found": channels,
                "expected_channels_found": found_expected,
                "total_expected": len(expected_channels)
            }
        )
    
    def _try_parse_json(self, content: str) -> Tuple[Optional[dict], Optional[str]]:
        """Попытаться распарсить JSON несколькими стратегиями"""
        try:
            return json.loads(content), None
        except json.JSONDecodeError as err:
            try:
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    candidate = match.group(0)
                    try:
                        return json.loads(candidate), None
                    except json.JSONDecodeError as err_inner:
                        return None, f"Primary error: {err}; Extracted segment error: {err_inner}"
            except Exception as regex_err:
                self.log("WARN", f"Regex error in JSON parsing: {regex_err}")
            
            return None, f"JSON parse error: {err}; no JSON object found"
    
    def _analyze_harmony_channels(self, content: str) -> Dict[str, List[str]]:
        """Анализировать каналы в Harmony format ответе"""
        channels = {}
        channel_patterns = [
            r'<\|channel\|>(analysis|commentary|final)<\|message\|>([^<]*)',
            r'\*\*(Analysis|Commentary|Final)\*\*[:\s]*([^\n]*)',
            r'\[(analysis|commentary|final)\]([^\[]*)',
        ]
        
        for pattern in channel_patterns:
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        channel = match.group(1).lower()
                        message = match.group(2).strip()
                        if channel not in channels:
                            channels[channel] = []
                        channels[channel].append(message)
            except Exception as e:
                self.log("WARN", f"Channel pattern error: {e}")
                continue
        if not channels:
            if any(word in content.lower() for word in ['анализ', 'analysis', 'рассуждение', 'reasoning']):
                channels['analysis'] = [content[:200] + "..."]
        
        return channels
    
    def _analyze_reasoning_quality(self, content: str) -> Dict[str, Any]:
        """Анализировать качество reasoning в ответе"""
        analysis = {
            "step_count": 0,
            "has_conclusion": False,
            "has_mathematical_notation": False,
            "reasoning_depth": "low"
        }
        step_indicators = ["step", "этап", "во-первых", "во-вторых", "затем", "далее", "finally"]
        for indicator in step_indicators:
            try:
                pattern = rf'\b{re.escape(indicator)}\b'
                analysis["step_count"] += len(re.findall(pattern, content, re.IGNORECASE))
            except Exception:
                continue
        conclusion_indicators = ["conclusion", "заключение", "итак", "следовательно", "therefore"]
        analysis["has_conclusion"] = any(indicator in content.lower() for indicator in conclusion_indicators)
        math_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'=', r'\b\d+\b']
        try:
            analysis["has_mathematical_notation"] = any(re.search(pattern, content) for pattern in math_patterns)
        except Exception:
            analysis["has_mathematical_notation"] = False
        if analysis["step_count"] > 3 and analysis["has_conclusion"]:
            analysis["reasoning_depth"] = "high"
        elif analysis["step_count"] > 1:
            analysis["reasoning_depth"] = "medium"
        
        return analysis

