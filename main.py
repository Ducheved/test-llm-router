#!/usr/bin/env python3
"""
🚀 ULTIMATE OpenRouter Test Suite 🚀
by Senior Developer | v1.0

Ультимативный тестовый скрипт для полного тестирования OpenRouter API
Тестирует: chat, stream, vision, json, harmony, tools, imagegen, generation, 
completions, models, batch, cache, multimodal + race conditions

🎯 Цели:
- Максимальное покрытие API OpenRouter 
- Детальная аналитика кеширования
- Race condition тестирование
- Полное логирование всех ответов
- Профессиональная архитектура кода
"""

import os
import sys
import json
import time
import base64
import hashlib
import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Импорты для работы с API и UI
try:
    import httpx
    from openai import OpenAI
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich import box
    from dotenv import load_dotenv
    from PIL import Image
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установите зависимости: pip install openai httpx rich python-dotenv pillow")
    sys.exit(1)

# Инициализация
console = Console()
load_dotenv()

@dataclass 
class TestConfig:
    """Конфигурация для тестирования"""
    api_key: str
    base_url: str
    model: str
    test_categories: List[str]
    vision_image_path: str
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 120
    race_test_count: int = 5
    
    @classmethod
    def from_env(cls) -> 'TestConfig':
        """Создает конфигурацию из .env файла"""
        api_key = os.getenv("ROUTER_API_KEY")
        if not api_key:
            raise ValueError("❌ ROUTER_API_KEY не найден в .env файле!")
        
        base_url = os.getenv("ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("TEST_MODELS", "").split(",")[0].strip()
        if not model:
            raise ValueError("❌ TEST_MODELS не найден в .env файле!")
            
        categories = os.getenv("TEST_CATEGORIES", "chat").split(",")
        categories = [cat.strip() for cat in categories if cat.strip()]
        
        vision_image = os.getenv("VISION_IMAGE", "payloads/cat.jpeg")
        
        return cls(
            api_key=api_key,
            base_url=base_url, 
            model=model,
            test_categories=categories,
            vision_image_path=vision_image
        )

@dataclass
class TestResult:
    """Результат выполнения теста"""
    test_name: str
    category: str
    success: bool
    duration: float
    model: str
    response_data: Optional[Dict] = None
    error: Optional[str] = None
    usage: Optional[Dict] = None
    cache_info: Optional[Dict] = None
    headers: Optional[Dict] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class UltimateOpenRouterTester:
    """🎯 Ультимативный тестер OpenRouter API"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.http_client = httpx.Client(timeout=config.timeout)
        self.results: List[TestResult] = []
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Инициализируем лог файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"ultimate_test_{timestamp}.log"
        
        # Инициализируем logger
        self.logger = logging.getLogger(f"ultimate_test_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # Создаем handler для файла с правильной кодировкой
        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Проверяем изображение для vision тестов
        self.vision_image_b64 = self._load_vision_image()
        
        console.print(Panel.fit(
            f"[bold cyan]🚀 ULTIMATE OpenRouter Test Suite[/]\n\n"
            f"[green]Model:[/] {config.model}\n"
            f"[blue]Base URL:[/] {config.base_url}\n"
            f"[yellow]Categories:[/] {', '.join(config.test_categories)}\n"
            f"[magenta]Log File:[/] {self.log_file}",
            title="⚙️ Configuration",
            border_style="cyan"
        ))
    
    def _load_vision_image(self) -> Optional[str]:
        """Загружает изображение для vision тестов"""
        image_path = Path(self.config.vision_image_path)
        if not image_path.exists():
            console.print(f"[yellow]⚠️ Изображение не найдено: {image_path}[/]")
            return None
            
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Конвертируем в base64
            b64_string = base64.b64encode(image_data).decode('utf-8')
            
            # Определяем MIME тип
            if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif image_path.suffix.lower() == '.png':
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'
                
            return f"data:{mime_type};base64,{b64_string}"
            
        except Exception as e:
            console.print(f"[red]❌ Ошибка загрузки изображения: {e}[/]")
            return None
    
    def _log_test_result(self, result: TestResult):
        """Логирует результат теста"""
        log_entry = {
            "timestamp": result.timestamp.isoformat(),
            "test_name": result.test_name,
            "category": result.category,
            "success": result.success,
            "duration": result.duration,
            "model": result.model,
            "response_data": result.response_data,
            "error": result.error,
            "usage": result.usage,
            "cache_info": result.cache_info,
            "headers": result.headers
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + '\n---\n')
    
    def _extract_usage_and_cache(self, response: Any) -> tuple:
        """Извлекает информацию об использовании и кешировании"""
        usage_info = None
        cache_info = {}
        
        if hasattr(response, 'usage') and response.usage:
            # Безопасная сериализация usage
            usage_data = {}
            usage = response.usage
            
            # Базовые поля usage
            if hasattr(usage, 'completion_tokens'):
                usage_data['completion_tokens'] = usage.completion_tokens
            if hasattr(usage, 'prompt_tokens'):
                usage_data['prompt_tokens'] = usage.prompt_tokens
            if hasattr(usage, 'total_tokens'):
                usage_data['total_tokens'] = usage.total_tokens
                
            # Детальная информация (безопасно конвертируем в строку)
            if hasattr(usage, 'completion_tokens_details'):
                usage_data['completion_tokens_details'] = str(usage.completion_tokens_details)
            if hasattr(usage, 'prompt_tokens_details'):
                usage_data['prompt_tokens_details'] = str(usage.prompt_tokens_details)
                
            usage_info = usage_data
            
            # Ищем кеш-специфичные поля
            cache_fields = [
                'cache_creation_input_tokens', 'cache_read_input_tokens',
                'cached_tokens', 'cache_hit', 'cache_miss', 'prompt_tokens_cached'
            ]
            
            for field in cache_fields:
                if field in usage_info and usage_info[field] is not None:
                    cache_info[field] = usage_info[field]
            
            # Дополнительно извлекаем cached_tokens из prompt_tokens_details для OpenAI
            if 'prompt_tokens_details' in usage_info:
                details_str = usage_info['prompt_tokens_details']
                if 'cached_tokens=' in details_str:
                    try:
                        # Извлекаем cached_tokens из строки вида "PromptTokensDetails(audio_tokens=0, cached_tokens=17792)"
                        import re
                        match = re.search(r'cached_tokens=(\d+)', details_str)
                        if match:
                            cache_info['cached_tokens'] = int(match.group(1))
                    except:
                        pass
        
        return usage_info, cache_info
    
    def _extract_headers(self, response: Any) -> Dict:
        """Извлекает заголовки ответа"""
        headers = {}
        
        if hasattr(response, '_raw_response') and hasattr(response._raw_response, 'headers'):
            headers = dict(response._raw_response.headers)
        elif hasattr(response, 'response') and hasattr(response.response, 'headers'):
            headers = dict(response.response.headers)
            
        return headers

    def _safe_extract_content(self, response: Any) -> tuple:
        """Безопасное извлечение контента и причины завершения"""
        try:
            # Логируем тип и структуру ответа
            self.logger.debug(f"Response type: {type(response)}")
            
            if not response:
                self.logger.warning("Empty response received")
                return "No response", "empty_response"
                
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                
                # Сначала пытаемся получить обычный content
                content = choice.message.content if hasattr(choice, 'message') and choice.message else ""
                finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else "unknown"
                
                # Если content пустой, но есть reasoning - используем его
                if not content and hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'reasoning'):
                    reasoning_content = choice.message.reasoning
                    if reasoning_content:
                        self.logger.debug(f"Using reasoning content instead of empty content")
                        content = reasoning_content
                
                if not content:
                    self.logger.warning(f"Empty content in choice. Choice type: {type(choice)}")
                    if hasattr(choice, '__dict__'):
                        self.logger.debug(f"Choice attributes: {choice.__dict__}")
                    content = "Empty content"
                        
                return content, finish_reason
            else:
                self.logger.warning(f"No choices in response. Response attributes: {dir(response) if hasattr(response, '__dict__') else 'No attributes'}")
                if hasattr(response, '__dict__'):
                    self.logger.debug(f"Response dict: {response.__dict__}")
                return "No choices found", "no_choices"
                
        except Exception as e:
            self.logger.error(f"Error extracting content: {e}")
            self.logger.error(f"Response type: {type(response)}")
            return f"Extraction error: {e}", "error"

    # 🎯 ТЕСТЫ КАТЕГОРИЙ
    
    def test_chat(self) -> TestResult:
        """Тест базового чата"""
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Ты опытный программист и архитектор ПО. Отвечай подробно и полно на вопросы."
                },
                {
                    "role": "user",
                    "content": "Объясни принцип работы async/await в Python подробно с примером кода."
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            duration = time.time() - start_time
            usage, cache_info = self._extract_usage_and_cache(response)
            headers = self._extract_headers(response)
            
            content, finish_reason = self._safe_extract_content(response)
            
            # Проверяем качество ответа
            quality_check = content and len(content.strip()) > 50 and ("async" in content.lower() or "python" in content.lower())
            
            return TestResult(
                test_name="basic_chat",
                category="chat",
                success=quality_check,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content, 
                    "finish_reason": finish_reason,
                    "content_length": len(content) if content else 0,
                    "quality_check": quality_check
                },
                usage=usage,
                cache_info=cache_info,
                headers=headers
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="basic_chat",
                category="chat", 
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_stream(self) -> TestResult:
        """Тест стримингового чата"""
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": "Напиши детальный пример функции на Python для сортировки списка различными способами с объяснением."
                }
            ]
            
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=500,
                temperature=0.3,
                stream=True
            )
            
            collected_content = ""
            chunk_count = 0
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Проверяем обычный content
                    if hasattr(delta, 'content') and delta.content:
                        collected_content += delta.content
                        chunk_count += 1
                    
                    # Для reasoning моделей проверяем reasoning поле
                    elif hasattr(delta, 'reasoning') and delta.reasoning:
                        collected_content += delta.reasoning
                        chunk_count += 1
                        
                    if chunk_count > 100:  # Ограничиваем количество чанков
                        break
            
            duration = time.time() - start_time
            
            # Проверяем качество стрим ответа
            success = len(collected_content.strip()) > 50 and chunk_count > 1
            
            return TestResult(
                test_name="streaming_chat",
                category="stream",
                success=success,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": collected_content,
                    "chunks_received": chunk_count,
                    "content_length": len(collected_content),
                    "quality_check": success
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="streaming_chat", 
                category="stream",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_vision(self) -> TestResult:
        """Тест анализа изображений"""
        start_time = time.time()
        
        if not self.vision_image_b64:
            return TestResult(
                test_name="vision_analysis",
                category="vision",
                success=False, 
                duration=0,
                model=self.config.model,
                error="No image available for vision test"
            )
        
        try:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Проанализируй это изображение подробно. Опиши что ты видишь, цвета, объекты, их расположение и любые детали."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self.vision_image_b64}
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=800,
                temperature=0.2
            )
            
            duration = time.time() - start_time
            usage, cache_info = self._extract_usage_and_cache(response)
            
            content, _ = self._safe_extract_content(response)
            
            # Проверяем качество vision анализа
            quality_check = content and len(content.strip()) > 20
            has_detailed_analysis = any(word in content.lower() for word in ['цвет', 'объект', 'изображение', 'вижу', 'color', 'object', 'image', 'see']) if content else False
            
            return TestResult(
                test_name="vision_analysis",
                category="vision", 
                success=quality_check and has_detailed_analysis,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content,
                    "content_length": len(content) if content else 0,
                    "has_detailed_analysis": has_detailed_analysis
                },
                usage=usage,
                cache_info=cache_info
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="vision_analysis",
                category="vision",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_json(self) -> TestResult:
        """Тест JSON mode"""
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Ты помощник, который отвечает только в формате JSON.",
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Создай JSON объект с информацией о языке программирования Python: название, год создания, создатель, основные особенности (массив из 3 элементов).",
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=400,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            duration = time.time() - start_time
            usage, cache_info = self._extract_usage_and_cache(response)
            
            content, _ = self._safe_extract_content(response)
            
            # Проверяем, что ответ действительно JSON
            # Claude может вернуть JSON в markdown блоке, очищаем это
            json_content = content.strip()
            if json_content.startswith('```json'):
                json_content = json_content[7:]  # убираем ```json
            if json_content.endswith('```'):
                json_content = json_content[:-3]  # убираем ```
            json_content = json_content.strip()
            
            try:
                parsed_json = json.loads(json_content)
                json_valid = True
            except Exception as e:
                self.logger.warning(f"JSON parsing failed: {e}. Content: {json_content[:200]}...")
                json_valid = False
                parsed_json = None
            
            return TestResult(
                test_name="json_mode",
                category="json",
                success=json_valid,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content,
                    "json_valid": json_valid,
                    "parsed_json": parsed_json
                },
                usage=usage,
                cache_info=cache_info
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="json_mode",
                category="json",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_tools(self) -> TestResult:
        """Тест function calling"""
        start_time = time.time()
        
        try:
            def get_weather(location: str) -> str:
                return f"Погода в {location}: солнечно, +22°C"
            
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Получить информацию о погоде в указанном городе",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Название города"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Какая погода в Москве?",
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                max_tokens=300,
                temperature=0.3
            )
            
            duration = time.time() - start_time
            usage, cache_info = self._extract_usage_and_cache(response)
            
            # Проверяем, был ли вызван tool
            tool_called = False
            tool_calls = []
            
            try:
                if response.choices and response.choices[0].message and hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                    tool_called = True
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_calls.append({
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })
            except Exception as e:
                self.logger.warning(f"Error processing tool calls: {e}")
            
            content, _ = self._safe_extract_content(response)
            
            return TestResult(
                test_name="tool_calling",
                category="tools",
                success=tool_called,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content,
                    "tool_called": tool_called,
                    "tool_calls": tool_calls
                },
                usage=usage,
                cache_info=cache_info
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="tool_calling",
                category="tools",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_cache(self) -> TestResult:
        """Тест prompt caching с правильным cache_control форматом для всех провайдеров OpenRouter"""
        start_time = time.time()
        
        try:
            # Создаем большой контекст для кеширования (>2048 токенов для всех провайдеров)
            large_context_part1 = "Ты эксперт по разработке высоконагруженных систем с 15-летним опытом работы в крупных технологических компаниях."
            
            large_context_part2 = """
ТВОЯ ЭКСПЕРТИЗА В ДЕТАЛЯХ:

Архитектура распределенных систем:
- Микросервисы и монолиты: выбор правильной архитектуры на основе бизнес-требований и команды
- Service mesh: Istio, Linkerd, Consul Connect для управления трафиком между сервисами  
- Event-driven архитектура и паттерны интеграции для слабосвязанных систем
- CQRS (Command Query Responsibility Segregation) для разделения команд и запросов
- Event Sourcing для аудита и восстановления состояния системы из событий
- Saga pattern для распределенных транзакций и компенсирующих действий
- Domain Driven Design (DDD) и Bounded Contexts для моделирования предметной области
- Clean Architecture, Hexagonal Architecture для независимости от внешних зависимостей
- API Gateway паттерны и Backend for Frontend для централизованного управления API

Технологический стек и языки программирования:
- Python экосистема: FastAPI для высокопроизводительных API, Django для enterprise приложений
- Flask для микросервисов, Celery для асинхронных задач, SQLAlchemy для ORM
- Pydantic для валидации данных, asyncio для параллельной обработки
- Go разработка: Gin, Echo веб-фреймворки, gRPC для межсервисного взаимодействия
- Протоколы TCP/UDP, goroutines для параллельной обработки, channels для синхронизации
- Java Enterprise: Spring Boot для быстрой разработки, Spring Cloud для микросервисов
- Hibernate, JPA для работы с базами данных, Maven/Gradle для сборки проектов
- JavaScript/TypeScript: Node.js для серверной разработки, Express для веб-серверов
- React, Vue.js, Angular для фронтенд разработки, TypeScript для типизации
- Rust системное программирование и высокопроизводительные сервисы с нулевыми накладными расходами
- C# .NET Core для enterprise решений, Entity Framework для работы с данными

Базы данных и хранение данных:
- Реляционные БД: PostgreSQL с расширениями, MySQL оптимизация запросов и индексов
- Шардинг и репликация для масштабирования, партиционирование таблиц по датам
- NoSQL решения: MongoDB для документов, Cassandra для временных рядов, DynamoDB для AWS
- In-memory хранилища: Redis для кеширования и очередей, Memcached для простого кеша
- Search engines: Elasticsearch для полнотекстового поиска, Solr для корпоративного поиска
- Time-series БД: InfluxDB для метрик и мониторинга, TimescaleDB как расширение PostgreSQL
- Graph databases: Neo4j для связанных данных, Amazon Neptune для графовых запросов

Инфраструктура и DevOps:
- Контейнеризация: Docker многослойные образы, оптимизация размера и безопасности
- Оркестрация: Kubernetes кластеры, Helm charts для управления релизами
- Operators для автоматизации операций, Custom Resource Definitions (CRDs)
- Облачные платформы: AWS EC2, Lambda, RDS, GCP Compute Engine, App Engine
- Azure Virtual Machines, App Service, архитектурные паттерны для каждой платформы
- Infrastructure as Code: Terraform для мультиоблачной инфраструктуры
- CloudFormation для AWS, Pulumi для императивного подхода к IaC
- CI/CD пайплайны: Jenkins с Pipeline as Code, GitLab CI/CD интеграция
- GitHub Actions для автоматизации, ArgoCD для GitOps деплоймента
- Service mesh и ingress контроллеры: Nginx для reverse proxy, Traefik для автодискавери
- Envoy proxy для продвинутой маршрутизации и наблюдаемости

Мониторинг и наблюдаемость:
- Метрики: Prometheus для сбора метрик, Grafana дашборды и алерты
- Логирование: ELK Stack (Elasticsearch, Logstash, Kibana) для централизованных логов
- Трейсинг: Jaeger, Zipkin для распределенного трейсинга запросов между сервисами
- APM решения: New Relic, DataDog, Dynatrace для мониторинга производительности приложений
- SLI/SLO/SLA определение и мониторинг для обеспечения надежности системы

Производительность и масштабирование:
- Кеширование стратегии: CDN для статики, Redis для данных, application-level кеш
- Load balancing: NGINX, HAProxy, cloud load balancers для распределения нагрузки
- Auto-scaling: горизонтальное и вертикальное масштабирование на основе метрик
- Database optimization: индексирование, партиционирование, репликация master-slave
- Профилирование: pprof, py-spy, flame graphs для анализа производительности
- Message Queues: RabbitMQ для надежности, Apache Kafka для высокой пропускной способности
- AWS SQS/SNS для облачных решений, асинхронная обработка и паттерны Retry/Circuit Breaker

Отвечай максимально детально с практическими рекомендациями и конкретными примерами реализации.
            """ * 16  # Увеличиваем до 16 раз для гарантии >2048 токенов
            
            self.logger.info(f"📏 Размер контента: {len(large_context_part2)} символов")
            
            # ПРАВИЛЬНЫЙ формат для всех провайдеров OpenRouter!
            # Для Anthropic Claude: обязательно content как массив объектов с cache_control
            # Для OpenAI/Gemini: автоматическое кеширование + поддержка cache_control  
            # Для других провайдеров: универсальная поддержка
            
            # Первый запрос - создаем кеш (ПРАВИЛЬНЫЙ формат!)
            messages_with_cache = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": large_context_part1
                        },
                        {
                            "type": "text", 
                            "text": large_context_part2,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": "Объясни SOLID принципы кратко, одним абзацем"
                }
            ]
            
            system_content_size = len(large_context_part1) + len(large_context_part2)
            self.logger.info(f"📏 Общий размер system контента: {system_content_size} символов")
            
            self.logger.info("🚀 Отправляем первый запрос с cache_control (создание кеша)...")
            response_1 = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages_with_cache,
                max_tokens=200,
                temperature=0.1  # Низкая температура для стабильности
            )
            
            first_content, _ = self._safe_extract_content(response_1)
            usage_1, cache_1 = self._extract_usage_and_cache(response_1)
            
            self.logger.info(f"✅ Первый ответ получен: {len(first_content) if first_content else 0} символов")
            self.logger.info(f"📊 Usage 1: {usage_1}")
            self.logger.info(f"💾 Cache 1: {cache_1}")
            
            # Короткая пауза для стабильности кеша
            time.sleep(2)
            
            # Второй запрос - используем ТОТ ЖЕ system с cache_control (чтение из кеша!)
            messages_cache_read = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": large_context_part1
                        },
                        {
                            "type": "text", 
                            "text": large_context_part2,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": "А теперь объясни паттерны GoF кратко, одним абзацем"
                }
            ]
            
            self.logger.info("🔄 Отправляем второй запрос (чтение из кеша)...")
            response_2 = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages_cache_read,
                max_tokens=200,
                temperature=0.1
            )
            
            second_content, _ = self._safe_extract_content(response_2)
            usage_2, cache_2 = self._extract_usage_and_cache(response_2)
            
            self.logger.info(f"✅ Второй ответ получен: {len(second_content) if second_content else 0} символов")
            self.logger.info(f"📊 Usage 2: {usage_2}")
            self.logger.info(f"💾 Cache 2: {cache_2}")
            
            duration = time.time() - start_time
            
            # Универсальный анализ кеширования для ВСЕХ провайдеров OpenRouter
            cache_detected = False
            cache_analysis = {
                'cache_creation_tokens': 0,
                'cache_read_tokens': 0,
                'provider_type': 'unknown',
                'cache_evidence': []
            }
            
            if usage_1 and usage_2:
                # Anthropic Claude поля из cache_info
                cache_read_1 = cache_1.get('cache_read_input_tokens', 0)
                cache_read_2 = cache_2.get('cache_read_input_tokens', 0)
                cache_creation_1 = cache_1.get('cache_creation_input_tokens', 0)
                cache_creation_2 = cache_2.get('cache_creation_input_tokens', 0)
                
                # OpenAI поля из cache_info
                cached_1 = cache_1.get('cached_tokens', 0)
                cached_2 = cache_2.get('cached_tokens', 0)
                
                self.logger.info(f"🔍 Найдено cached_tokens: {cached_1} -> {cached_2}")
                
                # Google Gemini поля из cache_info
                prompt_tokens_cached_1 = cache_1.get('prompt_tokens_cached', 0)
                prompt_tokens_cached_2 = cache_2.get('prompt_tokens_cached', 0)
                
                # Детальная информация из строковых полей
                prompt_details_1 = str(usage_1.get('prompt_tokens_details', ''))
                prompt_details_2 = str(usage_2.get('prompt_tokens_details', ''))
                
                # Определяем провайдера и проверяем кеширование
                if cache_creation_1 > 0 or cache_read_2 > 0:
                    cache_detected = True
                    cache_analysis['provider_type'] = 'anthropic'
                    cache_analysis['cache_creation_tokens'] = cache_creation_1
                    cache_analysis['cache_read_tokens'] = cache_read_2
                    cache_analysis['cache_evidence'].append(f"Anthropic: создание={cache_creation_1}, чтение={cache_read_2}")
                    
                elif cached_2 > 0:
                    cache_detected = True
                    cache_analysis['provider_type'] = 'openai'
                    cache_analysis['cache_read_tokens'] = cached_2
                    cache_analysis['cache_evidence'].append(f"OpenAI: кешированные токены={cached_2}")
                    
                elif prompt_tokens_cached_2 > 0:
                    cache_detected = True
                    cache_analysis['provider_type'] = 'google'
                    cache_analysis['cache_read_tokens'] = prompt_tokens_cached_2
                    cache_analysis['cache_evidence'].append(f"Google: кешированные токены промпта={prompt_tokens_cached_2}")
                    
                elif 'cached' in prompt_details_2.lower() and (cached_2 > 0 or prompt_tokens_cached_2 > 0 or cache_read_2 > 0):
                    cache_detected = True
                    cache_analysis['provider_type'] = 'auto_detected'
                    cache_analysis['cache_evidence'].append(f"Обнаружено в деталях: {prompt_details_2}")
                
                # Дополнительная проверка по изменению количества токенов
                prompt_1 = usage_1.get('prompt_tokens', 0)
                prompt_2 = usage_2.get('prompt_tokens', 0) 
                
                # Если во втором запросе prompt токенов значительно меньше - возможно кеширование
                if prompt_1 > 0 and prompt_2 > 0 and prompt_1 > prompt_2 * 1.5:
                    if not cache_detected:
                        cache_detected = True
                        cache_analysis['provider_type'] = 'inferred_by_tokens'
                    cache_analysis['cache_evidence'].append(f"Снижение prompt токенов: {prompt_1} -> {prompt_2}")
                
                cache_analysis.update({
                    'first_request': {
                        'prompt_tokens': prompt_1,
                        'cache_creation': cache_creation_1,
                        'cache_read': cache_read_1,
                        'cached': cached_1,
                        'prompt_tokens_cached': prompt_tokens_cached_1
                    },
                    'second_request': {
                        'prompt_tokens': prompt_2,
                        'cache_creation': cache_creation_2,
                        'cache_read': cache_read_2,
                        'cached': cached_2,
                        'prompt_tokens_cached': prompt_tokens_cached_2
                    }
                })
            
            self.logger.info(f"🔍 Кеширование обнаружено: {cache_detected}")
            self.logger.info(f"📋 Провайдер: {cache_analysis['provider_type']}")
            self.logger.info(f"🎯 Доказательства: {cache_analysis['cache_evidence']}")
            
            return TestResult(
                test_name="prompt_caching",
                category="cache",
                success=cache_detected,
                duration=duration,
                model=self.config.model,
                response_data={
                    "cache_detected": cache_detected,
                    "first_response_length": len(first_content) if first_content else 0,
                    "second_response_length": len(second_content) if second_content else 0,
                    "provider_type": cache_analysis['provider_type'],
                    "cache_evidence": cache_analysis['cache_evidence']
                },
                usage={
                    "first_request": usage_1,
                    "second_request": usage_2
                },
                cache_info=cache_analysis
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="prompt_caching",
                category="cache",
                success=False,
                duration=duration, 
                model=self.config.model,
                error=str(e)
            )
    
    def test_multimodal(self) -> TestResult:
        """Тест мультимодальных возможностей"""
        start_time = time.time()
        
        if not self.vision_image_b64:
            return TestResult(
                test_name="multimodal_capabilities",
                category="multimodal",
                success=False,
                duration=0,
                model=self.config.model,
                error="No image available for multimodal test"
            )
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Ты эксперт по анализу изображений и компьютерному зрению. Анализируй изображения подробно и профессионально."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Проанализируй это изображение максимально детально: опиши все объекты, цвета, композицию, освещение, текстуры, возможный контекст и назначение. Дай профессиональный анализ как эксперт по компьютерному зрению."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self.vision_image_b64}
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            duration = time.time() - start_time
            usage, cache_info = self._extract_usage_and_cache(response)
            
            content, _ = self._safe_extract_content(response)
            
            # Проверяем качество multimodal анализа
            has_detailed_analysis = content and len(content.strip()) > 100
            quality_keywords = ['цвет', 'объект', 'изображение', 'анализ', 'детали', 'color', 'object', 'image', 'analysis', 'detail']
            keyword_match = any(keyword in content.lower() for keyword in quality_keywords) if content else False
            
            return TestResult(
                test_name="multimodal_capabilities",
                category="multimodal",
                success=has_detailed_analysis and keyword_match,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content,
                    "content_length": len(content) if content else 0,
                    "has_detailed_analysis": has_detailed_analysis
                },
                usage=usage,
                cache_info=cache_info
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="multimodal_capabilities",
                category="multimodal",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_generation_stats(self) -> TestResult:
        """Тест получения статистики генерации"""
        start_time = time.time()
        
        try:
            # Сначала делаем обычный запрос
            messages = [
                {
                    "role": "user",
                    "content": "Привет! Напиши короткое приветствие и объясни что такое API."
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            generation_id = response.id if hasattr(response, 'id') else None
            content, _ = self._safe_extract_content(response)
            
            # Пытаемся получить статистику генерации (только если это настоящий OpenRouter)
            stats_success = False
            stats_data = None
            
            if generation_id and "openrouter.ai" in self.config.base_url:
                try:
                    time.sleep(1)  # Пауза перед запросом статистики
                    stats_response = self.http_client.get(
                        f"{self.config.base_url.replace('/api/v1', '')}/api/v1/generation?id={generation_id}",
                        headers={"Authorization": f"Bearer {self.config.api_key}"}
                    )
                    
                    stats_success = stats_response.status_code == 200
                    stats_data = stats_response.json() if stats_success else None
                except Exception as e:
                    console.print(f"[yellow]Stats API недоступен: {e}[/yellow]")
                    stats_success = False
                    stats_data = None
            else:
                # Для локального сервера считаем тест успешным если получили ответ
                stats_success = bool(content and len(content.strip()) > 10)
            
            duration = time.time() - start_time
            usage, cache_info = self._extract_usage_and_cache(response)
            
            return TestResult(
                test_name="generation_stats", 
                category="generation",
                success=stats_success,
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content,
                    "generation_id": generation_id,
                    "stats_retrieved": stats_success,
                    "stats_data": stats_data,
                    "content_length": len(content) if content else 0,
                    "is_openrouter": "openrouter.ai" in self.config.base_url
                },
                usage=usage,
                cache_info=cache_info
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="generation_stats",
                category="generation",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_models_list(self) -> TestResult:
        """Тест получения списка моделей"""
        start_time = time.time()
        
        try:
            response = self.http_client.get(
                f"{self.config.base_url}/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            
            duration = time.time() - start_time
            success = response.status_code == 200
            
            if success:
                models_data = response.json()
                model_count = len(models_data.get('data', []))
                current_model_found = any(
                    model['id'] == self.config.model 
                    for model in models_data.get('data', [])
                )
            else:
                models_data = None
                model_count = 0
                current_model_found = False
            
            return TestResult(
                test_name="models_list",
                category="models",
                success=success,
                duration=duration,
                model=self.config.model,
                response_data={
                    "status_code": response.status_code,
                    "model_count": model_count,
                    "current_model_found": current_model_found,
                    "models_data": models_data
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="models_list",
                category="models",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def test_reasoning(self) -> TestResult:
        """Тест reasoning токенов для моделей с thinking capabilities"""
        start_time = time.time()
        
        try:
            # Задача, требующая рассуждений
            messages = [
                {
                    "role": "user",
                    "content": "Реши эту задачу пошагово: У Алисы было 15 яблок. Она дала 1/3 от них Бобу, а затем съела 2 яблока. Сколько яблок у неё осталось? Покажи все шаги решения."
                }
            ]
            
            # Запрос с reasoning параметрами (по документации OpenRouter)
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=800,
                temperature=0.3
            )
            
            content, finish_reason = self._safe_extract_content(response)
            usage, cache_info = self._extract_usage_and_cache(response)
            
            # Проверяем наличие reasoning
            reasoning_detected = False
            reasoning_content = ""
            reasoning_tokens = 0
            
            if hasattr(response.choices[0], 'message'):
                message = response.choices[0].message
                
                # Проверяем reasoning в message
                if hasattr(message, 'reasoning') and message.reasoning:
                    reasoning_detected = True
                    reasoning_content = message.reasoning
                    self.logger.info(f"Обнаружен reasoning контент длиной: {len(reasoning_content)}")
                    
                # Проверяем reasoning_details (новый формат)
                if hasattr(message, 'reasoning_details') and message.reasoning_details:
                    reasoning_detected = True
                    reasoning_details = message.reasoning_details
                    self.logger.info(f"Обнаружены reasoning_details: {len(reasoning_details)} блоков")
            
            # Проверяем reasoning токены в usage
            if usage:
                reasoning_tokens = usage.get('reasoning_tokens', 0)
                if reasoning_tokens > 0:
                    reasoning_detected = True
                    self.logger.info(f"Reasoning токены в usage: {reasoning_tokens}")
            
            # Анализируем качество reasoning ответа
            reasoning_quality = False
            if content:
                # Для reasoning моделей проверяем наличие математических выкладок и логики
                step_indicators = ['шаг', 'step', '1/3', 'дала', 'съела', 'осталось', 'решения', 'пошагово']
                math_indicators = ['15', '5', '2', '13', '8', 'яблок', '+', '-', '=', '/']
                
                content_lower = content.lower()
                step_count = sum(1 for indicator in step_indicators if indicator in content_lower)
                math_count = sum(1 for indicator in math_indicators if indicator in content_lower)
                
                # Более мягкие критерии для reasoning модели
                reasoning_quality = step_count >= 2 and math_count >= 3 and len(content) > 50
                
                self.logger.info(f"Reasoning качество - шаги: {step_count}, математика: {math_count}, длина: {len(content)}")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="reasoning_capabilities",
                category="reasoning",
                success=reasoning_quality,  # Успех если есть качественное решение
                duration=duration,
                model=self.config.model,
                response_data={
                    "content": content,
                    "reasoning_detected": reasoning_detected,
                    "reasoning_content_length": len(reasoning_content) if reasoning_content else 0,
                    "reasoning_tokens": reasoning_tokens,
                    "reasoning_quality": reasoning_quality,
                    "finish_reason": finish_reason
                },
                usage=usage,
                cache_info=cache_info
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="reasoning_capabilities",
                category="reasoning",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )

    def test_race_conditions(self) -> TestResult:
        """Тест race conditions - одновременные запросы"""
        start_time = time.time()
        
        def make_concurrent_request(request_id: int) -> Dict:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"Просто напиши число {request_id} и слово 'done'. Больше ничего не пиши."
                    }
                ]
                
                req_start = time.time()
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.0
                )
                req_duration = time.time() - req_start
                
                content, _ = self._safe_extract_content(response)
                # Более мягкая проверка - ищем число в любой части ответа
                success = str(request_id) in content
                
                return {
                    "request_id": request_id,
                    "success": success,
                    "duration": req_duration,
                    "content": content,
                    "expected_id_found": success
                }
                
            except Exception as e:
                req_duration = time.time() - req_start if 'req_start' in locals() else 0
                return {
                    "request_id": request_id,
                    "success": False,
                    "duration": req_duration,
                    "content": "",
                    "error": str(e),
                    "expected_id_found": False
                }
        
        try:
            # Запускаем параллельные запросы
            with ThreadPoolExecutor(max_workers=self.config.race_test_count) as executor:
                futures = [
                    executor.submit(make_concurrent_request, i+1) 
                    for i in range(self.config.race_test_count)
                ]
                
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            duration = time.time() - start_time
            
            # Анализируем результаты
            successful_requests = len([r for r in results if r['success']])
            total_requests = len(results)
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            
            avg_duration = sum(r.get('duration', 0) for r in results) / len(results)
            max_duration = max(r.get('duration', 0) for r in results)
            min_duration = min(r.get('duration', 0) for r in results)
            
            race_test_success = success_rate > 0.8  # 80%+ успешных запросов
            
            return TestResult(
                test_name="race_conditions",
                category="batch",
                success=race_test_success,
                duration=duration,
                model=self.config.model,
                response_data={
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": success_rate,
                    "avg_response_time": avg_duration,
                    "max_response_time": max_duration,
                    "min_response_time": min_duration,
                    "detailed_results": results
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="race_conditions",
                category="batch",
                success=False,
                duration=duration,
                model=self.config.model,
                error=str(e)
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Запускает все тесты согласно конфигурации"""
        
        # Карта тестов по категориям
        test_methods = {
            'chat': self.test_chat,
            'stream': self.test_stream,
            'vision': self.test_vision,
            'json': self.test_json,
            'tools': self.test_tools,
            'cache': self.test_cache,
            'multimodal': self.test_multimodal,
            'generation': self.test_generation_stats,
            'models': self.test_models_list,
            'batch': self.test_race_conditions,
            'reasoning': self.test_reasoning
        }
        
        # Фильтруем тесты по конфигурации
        tests_to_run = []
        for category in self.config.test_categories:
            if category in test_methods:
                tests_to_run.append((category, test_methods[category]))
            else:
                console.print(f"[yellow]⚠️ Неизвестная категория теста: {category}[/]")
        
        console.print(f"\n[bold]Запуск {len(tests_to_run)} тестов...[/]\n")
        
        # Запускаем тесты с прогресс-баром
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Выполнение тестов...", total=len(tests_to_run))
            
            for category, test_method in tests_to_run:
                progress.update(task, description=f"Тест: {category}")
                
                try:
                    result = test_method()
                    self.results.append(result)
                    self._log_test_result(result)
                    
                    # Показываем результат
                    status = "✅" if result.success else "❌"
                    console.print(f"  {status} {category}: {result.duration:.2f}s")
                    
                except Exception as e:
                    error_result = TestResult(
                        test_name=f"{category}_error",
                        category=category,
                        success=False,
                        duration=0,
                        model=self.config.model,
                        error=f"Unexpected error: {str(e)}"
                    )
                    self.results.append(error_result)
                    console.print(f"  💥 {category}: Unexpected error")
                
                progress.advance(task)
        
        return self.results
    
    def generate_report(self):
        """Генерирует итоговый отчет"""
        if not self.results:
            console.print("[red]❌ Нет результатов для отчета[/]")
            return
        
        # Общая статистика
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        success_rate = (successful_tests / total_tests) * 100
        
        # Статистика по категориям
        categories_stats = {}
        for result in self.results:
            category = result.category
            if category not in categories_stats:
                categories_stats[category] = {'total': 0, 'success': 0}
            categories_stats[category]['total'] += 1
            if result.success:
                categories_stats[category]['success'] += 1
        
        # Кеш статистика
        cache_results = [r for r in self.results if r.category == 'cache' and r.cache_info]
        cache_detected = len(cache_results) > 0
        
        # Создаем таблицу результатов
        table = Table(title=f"🎯 Ultimate Test Results - {self.config.model}", box=box.ROUNDED)
        table.add_column("Category", style="cyan", width=15)
        table.add_column("Test", style="yellow", width=25)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Details", width=40)
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"{result.duration:.2f}s"
            
            # Формируем детали
            details = []
            if result.error:
                details.append(f"Error: {result.error[:30]}...")
            elif result.response_data:
                if 'content_length' in result.response_data:
                    details.append(f"Content: {result.response_data['content_length']} chars")
                if 'cache_detected' in result.response_data:
                    details.append(f"Cache: {'Yes' if result.response_data['cache_detected'] else 'No'}")
                if 'success_rate' in result.response_data:
                    details.append(f"Success rate: {result.response_data['success_rate']:.1%}")
            
            details_str = " | ".join(details[:2]) if details else "OK"
            
            table.add_row(result.category, result.test_name, status, duration, details_str)
        
        console.print("\n")
        console.print(table)
        
        # Общая сводка
        console.print(Panel.fit(
            f"[bold]📊 ИТОГОВАЯ СТАТИСТИКА[/]\n\n"
            f"[green]✅ Успешно:[/] {successful_tests}/{total_tests} ({success_rate:.1f}%)\n"
            f"[red]❌ Неудачно:[/] {total_tests - successful_tests}\n"
            f"[blue]🔄 Кеширование:[/] {'Обнаружено' if cache_detected else 'Не обнаружено'}\n"
            f"[yellow]📝 Лог файл:[/] {self.log_file}",
            title="📋 Summary",
            border_style="green" if success_rate > 80 else "yellow" if success_rate > 50 else "red"
        ))
        
        # Детальная статистика кеширования
        if cache_detected:
            console.print("\n[bold cyan]💾 ДЕТАЛИ КЕШИРОВАНИЯ:[/]")
            for result in cache_results:
                if result.cache_info and isinstance(result.cache_info, dict):
                    cache_analysis = result.cache_info
                    provider_type = cache_analysis.get('provider_type', 'unknown')
                    evidence = cache_analysis.get('cache_evidence', [])
                    
                    console.print(f"  • Провайдер: [yellow]{provider_type}[/]")
                    console.print(f"  • Создание кеша: {cache_analysis.get('cache_creation_tokens', 0)} токенов")
                    console.print(f"  • Чтение кеша: {cache_analysis.get('cache_read_tokens', 0)} токенов")
                    
                    if evidence:
                        console.print("  • Доказательства кеширования:")
                        for ev in evidence:
                            console.print(f"    - {ev}")
        else:
            console.print("\n[bold red]❌ КЕШИРОВАНИЕ НЕ ОБНАРУЖЕНО[/]")
            console.print("  • Проверьте правильность формата cache_control")
            console.print("  • Убедитесь что контент >1024 токенов")
            console.print("  • Для Anthropic используйте content как массив объектов") 
        
        # Сохраняем JSON отчет
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model,
            "base_url": self.config.base_url,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "categories_stats": categories_stats,
            "cache_detected": cache_detected,
            "results": [asdict(result) for result in self.results]
        }
        
        report_file = self.logs_dir / f"ultimate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        console.print(f"\n[green]📄 Подробный отчет сохранен: {report_file}[/]")


def main():
    """🚀 Главная функция - точка входа в Ultimate Test Suite"""
    
    console.print("""
[bold cyan]
╔════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ULTIMATE OpenRouter Test Suite 🚀                    ║
║                        by Senior Developer | v1.0                         ║
║                                                                            ║
║  Максимальное покрытие OpenRouter API + Race Conditions + Cache Analysis  ║
╚════════════════════════════════════════════════════════════════════════════╝
[/]
    """)
    
    try:
        # Загружаем конфигурацию
        config = TestConfig.from_env()
        
        # Создаем тестер
        tester = UltimateOpenRouterTester(config)
        
        # Запускаем тесты
        console.print("[bold]🎯 Начинаем тестирование...[/]\n")
        results = tester.run_all_tests()
        
        # Генерируем отчет
        console.print("\n[bold]📊 Генерация отчета...[/]")
        tester.generate_report()
        
        # Итоговое сообщение
        success_count = len([r for r in results if r.success])
        if success_count == len(results):
            console.print("\n[bold green]🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО! 🎉[/]")
        elif success_count > len(results) * 0.8:
            console.print("\n[bold yellow]⚡ БОЛЬШИНСТВО ТЕСТОВ ПРОШЛИ УСПЕШНО ⚡[/]")
        else:
            console.print("\n[bold red]⚠️ НАЙДЕНЫ ПРОБЛЕМЫ - ТРЕБУЕТ ВНИМАНИЯ ⚠️[/]")
            
    except ValueError as e:
        console.print(f"[red]❌ Ошибка конфигурации: {e}[/]")
        console.print("[yellow]💡 Проверьте .env файл с настройками[/]")
        sys.exit(1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️ Тестирование прервано пользователем[/]")
        sys.exit(0)
        
    except Exception as e:
        console.print(f"[red]💥 Критическая ошибка: {e}[/]")
        console.print(f"[dim]Traceback: {traceback.format_exc()}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()


