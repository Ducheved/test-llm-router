"""
OpenRouter API Test Suite - Core Test Infrastructure
Базовые классы для всех тестовых модулей
"""
import time
import json
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import httpx
except ImportError:
    httpx = None
    
from rich.console import Console

console = Console()

class TestStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Результат одного теста"""
    status: TestStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def is_success(self) -> bool:
        return self.status == TestStatus.SUCCESS

@dataclass 
class TestSuite:
    """Результаты набора тестов"""
    name: str
    results: Dict[str, TestResult]
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results.values() if r.is_success())
    
    @property 
    def total_count(self) -> int:
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

class BaseTestModule(ABC):
    """Базовый класс для всех тестовых модулей"""
    
    def __init__(self, config, http_client=None):
        self.config = config
        self.http = http_client
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Настроить логгер для модуля"""
        return console
    
    def log(self, level: str, message: str, error: Optional[Exception] = None):
        """Единообразное логирование"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        module_name = self.__class__.__name__
        
        color_map = {
            "DEBUG": "dim", "INFO": "cyan", "WARN": "yellow", 
            "ERROR": "red", "SUCCESS": "green"
        }
        color = color_map.get(level, "white")
        
        log_entry = f"[{timestamp}] [{module_name}] {message}"
        if error:
            log_entry += f" | {str(error)}"
        
        self.logger.print(f"[{color}]{log_entry}[/]")
    
    def retry_operation(self, operation, description: str = "operation"):
        """Выполнить операцию с повторами"""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.log("DEBUG", f"Попытка {attempt}/{self.config.max_retries}: {description}")
                return operation()
            except Exception as e:
                if attempt == self.config.max_retries:
                    self.log("ERROR", f"Все попытки исчерпаны для {description}", e)
                    raise
                self.log("WARN", f"Ошибка на попытке {attempt}: {e}")
                time.sleep(self.config.retry_delay * attempt)
    
    def time_operation(self, operation):
        """Измерить время выполнения операции"""
        start_time = time.time()
        try:
            result = operation()
            duration = time.time() - start_time
            return result, duration
        except Exception as e:
            duration = time.time() - start_time
            raise e
    
    @abstractmethod
    def get_test_methods(self) -> List[str]:
        """Вернуть список доступных методов тестирования"""
        pass
    
    @abstractmethod
    def run_test(self, test_name: str, model: str, **kwargs) -> TestResult:
        """Запустить конкретный тест"""
        pass

class APITestModule(BaseTestModule):
    """Базовый класс для модулей тестирования API"""
    
    def __init__(self, config, http_client=None):
        super().__init__(config, http_client)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openrouter-test-suite",
            "X-Title": "OpenRouter Test Suite"
        }
    
    def make_request(self, method: str, endpoint: str, **kwargs):
        """Сделать HTTP запрос к API"""
        url = self.config.get_api_url(endpoint)
        if 'headers' not in kwargs:
            kwargs['headers'] = self.headers
        else:
            kwargs['headers'] = {**self.headers, **kwargs['headers']}
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.config.timeout
        
        self.log("DEBUG", f"{method.upper()} {url}")
        
        if self.http and httpx:
            response = self.http.request(method, url, **kwargs)
        else:
            import requests
            response = requests.request(method, url, **kwargs)
        
        if response.status_code >= 400:
            self.log("WARN", f"HTTP {response.status_code}: {response.text[:200]}")
        
        return response

class ModelCapabilityDetector:
    """Класс для определения возможностей моделей через API"""
    
    def __init__(self, config, http_client=None):
        self.config = config
        self.http = http_client
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
        }
        self._models_cache = None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Получить информацию о модели через API"""
        if "/" not in model_id:
            return None
            
        author, slug = model_id.split("/", 1)
        url = self.config.get_api_url(f"models/{author}/{slug}")
        
        try:
            if self.http:
                response = self.http.get(url, headers=self.headers)
            else:
                import requests
                response = requests.get(url, headers=self.headers)
                
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        
        return None
    
    def is_multimodal(self, model_id: str, model_info: Optional[Dict] = None) -> bool:
        """Проверить поддерживает ли модель изображения"""
        if model_info:
            input_modalities = model_info.get("data", {}).get("architecture", {}).get("input_modalities", [])
            return "image" in input_modalities
        lower_model = model_id.lower()
        return any(keyword in lower_model for keyword in self.config.multimodal_keywords)
    
    def supports_reasoning(self, model_id: str, model_info: Optional[Dict] = None) -> bool:
        """Проверить поддерживает ли модель reasoning"""
        lower_model = model_id.lower()
        return any(keyword in lower_model for keyword in self.config.reasoning_models)
    
    def supports_image_generation(self, model_id: str, model_info: Optional[Dict] = None) -> bool:
        """Проверить поддерживает ли модель генерацию изображений"""
        if model_info:
            output_modalities = model_info.get("data", {}).get("architecture", {}).get("output_modalities", [])
            return "image" in output_modalities
        lower_model = model_id.lower()
        return any(keyword in lower_model for keyword in self.config.image_gen_keywords)

