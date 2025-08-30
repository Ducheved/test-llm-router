"""
OpenRouter Prompt Caching Test Module
Тестирование функции кеширования промптов OpenRouter API
https://openrouter.ai/docs/features/prompt-caching
"""
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..core import APITestModule, TestResult, TestStatus


class PromptCachingTestModule(APITestModule):
    """Тестирование кеширования промптов OpenRouter"""
    
    def __init__(self, config, http_client):
        super().__init__(config, http_client)
        if OpenAI:
            self.openai_client = OpenAI(
                api_key=config.api_key, 
                base_url=config.base_url
            )
        else:
            self.openai_client = None
        
        # Создаем отдельный лог файл для кеш-тестов
        self.cache_log_file = config.logs_dir / f"prompt_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        config.logs_dir.mkdir(exist_ok=True)
        
    def get_test_methods(self) -> List[str]:
        return [
            "cache_support_check",
            "cache_basic_test",
            "cache_repeated_requests", 
            "cache_system_message",
            "cache_different_models",
            "cache_long_conversation",
            "cache_token_analysis",
            "cache_historian_example"
        ]
    
    def _log_cache_response(self, test_name: str, model: str, request_data: Dict, response: Any, attempt: int = 1):
        """Логирует ответ в отдельный файл для анализа кеш-токенов"""
        timestamp = datetime.now().isoformat()
        
        # Создаем хеш запроса для идентификации
        request_hash = hashlib.md5(json.dumps(request_data, sort_keys=True).encode()).hexdigest()[:8]
        
        # Извлекаем заголовки более тщательно
        headers = {}
        if hasattr(response, '_raw_response'):
            if hasattr(response._raw_response, 'headers'):
                headers = dict(response._raw_response.headers)
        elif hasattr(response, 'response'):
            if hasattr(response.response, 'headers'):
                headers = dict(response.response.headers)
        elif hasattr(response, 'headers'):
            headers = dict(response.headers)
        
        log_entry = {
            "timestamp": timestamp,
            "test_name": test_name,
            "model": model,
            "attempt": attempt,
            "request_hash": request_hash,
            "request": request_data,
            "response": self._extract_response_info(response),
            "usage": getattr(response, 'usage', None).__dict__ if hasattr(response, 'usage') and response.usage else None,
            "headers": headers,
            "cache_info": self._extract_cache_info(response),
            "raw_response_type": type(response).__name__,
            "raw_response_attributes": [attr for attr in dir(response) if not attr.startswith('_')][:20]  # Первые 20 атрибутов для отладки
        }
        
        # Записываем в файл
        with open(self.cache_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + '\n---\n')
        
        self.log("INFO", f"Cache response logged to {self.cache_log_file}")
        
        # Дополнительно выводим информацию о найденных кеш-данных
        cache_info = self._extract_cache_info(response)
        if cache_info:
            self.log("SUCCESS", f"Found cache info: {cache_info}")
        else:
            self.log("WARN", f"No cache info found in response for {model}")
        
    def _extract_response_info(self, response: Any) -> Dict:
        """Извлекает основную информацию из ответа"""
        if hasattr(response, 'choices') and response.choices:
            return {
                "content": response.choices[0].message.content[:200] + "..." if len(response.choices[0].message.content) > 200 else response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason
            }
        return {"raw": str(response)[:200]}
    
    def _extract_cache_info(self, response: Any) -> Dict:
        """Извлекает информацию о кешировании из ответа"""
        cache_info = {}
        
        # Проверяем usage для cache tokens
        if hasattr(response, 'usage') and response.usage:
            usage_dict = response.usage.__dict__ if hasattr(response.usage, '__dict__') else {}
            
            # Ищем поля связанные с кешем
            for key, value in usage_dict.items():
                if 'cache' in key.lower() or 'cached' in key.lower():
                    cache_info[key] = value
        
        # Проверяем заголовки ответа - разные способы доступа
        headers_to_check = []
        
        # Пробуем разные способы получить headers
        if hasattr(response, '_raw_response') and hasattr(response._raw_response, 'headers'):
            headers_to_check = response._raw_response.headers
        elif hasattr(response, 'response') and hasattr(response.response, 'headers'):
            headers_to_check = response.response.headers  
        elif hasattr(response, 'headers'):
            headers_to_check = response.headers
        
        # Ищем заголовки связанные с кешированием
        if headers_to_check:
            for header_name, header_value in headers_to_check.items():
                header_lower = header_name.lower()
                if any(cache_word in header_lower for cache_word in ['cache', 'x-cache', 'cf-cache']):
                    cache_info[f"header_{header_name}"] = header_value
        
        # Проверяем модель ответа на дополнительные поля
        if hasattr(response, 'model_extra') and response.model_extra:
            for key, value in response.model_extra.items():
                if 'cache' in key.lower():
                    cache_info[key] = value
        
        # Если есть дополнительные поля в корневом объекте response
        if hasattr(response, '__dict__'):
            for key, value in response.__dict__.items():
                if 'cache' in key.lower() and key not in cache_info:
                    cache_info[key] = value
        
        return cache_info
    
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
            self.log("ERROR", f"Prompt caching test {test_name} failed for {model}", e)
            return TestResult(
                status=TestStatus.ERROR, 
                error=str(e)
            )
    
    def _test_cache_basic_test(self, model: str, **kwargs) -> TestResult:
        """Базовый тест кеширования - отправляем один и тот же запрос дважды"""
        self.log("INFO", f"Тестирование базового кеширования для {model}")
        
        if not self.openai_client:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="OpenAI client not available"
            )
        
        # Создаем запрос с cache_control согласно OpenRouter API
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Ты являешься экспертом по программированию с 15-летним опытом работы. Ты специализируешься на Python, JavaScript, и машинном обучении. Твоя задача - давать подробные, точные и практичные советы по программированию. Твои знания включают: современные фреймворки (React, Vue.js, FastAPI, Django), базы данных (PostgreSQL, MongoDB, Redis), облачные технологии (AWS, GCP, Azure), DevOps практики (Docker, Kubernetes, CI/CD), архитектурные паттерны (микросервисы, event-driven architecture), методологии разработки (Agile, TDD, DDD).",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user", 
                "content": "Объясни подробно концепцию кеширования в веб-приложениях, включая различные стратегии кеширования, их преимущества и недостатки."
            }
        ]
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.1  # Низкая температура для воспроизводимости
        }
        
        responses = []
        
        # Делаем первый запрос
        try:
            def make_request():
                return self.openai_client.chat.completions.create(**request_data)
            
            response1 = self.retry_operation(make_request, "first cache request")
            self._log_cache_response("cache_basic_test", model, request_data, response1, 1)
            responses.append({
                "attempt": 1,
                "content": response1.choices[0].message.content,
                "usage": response1.usage.__dict__ if response1.usage else {},
                "cache_info": self._extract_cache_info(response1)
            })
            
            # Небольшая пауза
            time.sleep(2)
            
            # Делаем второй запрос (должен использовать кеш)
            response2 = self.retry_operation(make_request, "second cache request")
            self._log_cache_response("cache_basic_test", model, request_data, response2, 2)
            responses.append({
                "attempt": 2,
                "content": response2.choices[0].message.content,
                "usage": response2.usage.__dict__ if response2.usage else {},
                "cache_info": self._extract_cache_info(response2)
            })
            
            self.log("SUCCESS", f"Базовый тест кеширования завершен для {model}")
            
            return TestResult(
                status=TestStatus.SUCCESS,
                data={
                    "responses": responses,
                    "cache_comparison": self._compare_cache_responses(responses),
                    "log_file": str(self.cache_log_file)
                }
            )
            
        except Exception as e:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Cache test failed: {str(e)}"
            )
    
    def _test_cache_repeated_requests(self, model: str, **kwargs) -> TestResult:
        """Тест множественных повторяющихся запросов для анализа кеширования"""
        self.log("INFO", f"Тестирование повторяющихся запросов для {model}")
        
        if not self.openai_client:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="OpenAI client not available"
            )
        
        # Длинный системный промпт для кеширования с cache_control
        system_message_content = [
            {
                "type": "text",
                "text": """Ты опытный архитектор программного обеспечения с глубокими знаниями в области:
        - Микросервисной архитектуры и паттернов интеграции
        - Облачных технологий (AWS, GCP, Azure) и serverless computing
        - Контейнеризации (Docker, Kubernetes) и оркестрации
        - CI/CD пайплайнов и DevOps практик
        - Реляционных и NoSQL баз данных (PostgreSQL, MongoDB, Redis)
        - Кеширования, CDN и оптимизации производительности
        - Безопасности приложений и защиты от уязвимостей
        - Мониторинга, логирования и observability
        - Event-driven архитектуры и message queues
        - Domain-Driven Design и чистой архитектуры
        
        Отвечай подробно, с примерами кода и архитектурными диаграммами где это уместно.""",
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": "Что такое микросервисная архитектура?"}
        ]
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.0
        }
        
        responses = []
        
        # Делаем 5 одинаковых запросов
        for i in range(5):
            try:
                def make_request():
                    return self.openai_client.chat.completions.create(**request_data)
                
                response = self.retry_operation(make_request, f"repeated request {i+1}")
                self._log_cache_response("cache_repeated_requests", model, request_data, response, i+1)
                
                responses.append({
                    "attempt": i+1,
                    "timestamp": datetime.now().isoformat(),
                    "content_length": len(response.choices[0].message.content),
                    "usage": response.usage.__dict__ if response.usage else {},
                    "cache_info": self._extract_cache_info(response)
                })
                
                # Пауза между запросами
                if i < 4:
                    time.sleep(1)
                    
            except Exception as e:
                self.log("ERROR", f"Repeated request {i+1} failed", e)
                responses.append({
                    "attempt": i+1,
                    "error": str(e)
                })
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "responses": responses,
                "cache_analysis": self._analyze_cache_patterns(responses),
                "log_file": str(self.cache_log_file)
            }
        )
    
    def _test_cache_system_message(self, model: str, **kwargs) -> TestResult:
        """Тест кеширования с фокусом на системные сообщения"""
        self.log("INFO", f"Тестирование кеширования системных сообщений для {model}")
        
        if not self.openai_client:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="OpenAI client not available"
            )
        
        # Длинный системный промпт для кеширования с cache_control
        long_system_message = [
            {
                "type": "text",
                "text": """Ты являешься экспертом в области разработки программного обеспечения с более чем 20-летним опытом работы в различных технологиях и отраслях. Твои компетенции включают:

ЯЗЫКИ ПРОГРАММИРОВАНИЯ:
- Python: Flask, Django, FastAPI, SQLAlchemy, Celery, NumPy, Pandas, Scikit-learn
- JavaScript/TypeScript: React, Vue.js, Angular, Node.js, Express, NestJS, Next.js
- Java: Spring Boot, Spring Framework, Hibernate, Maven, Gradle, JPA
- Go: Gin, Echo, gRPC, Docker, Kubernetes operators
- Rust: Tokio, Actix-Web, Serde, Warp, Rocket
- C#: .NET Core, Entity Framework, ASP.NET, Blazor, MAUI

АРХИТЕКТУРА И ПАТТЕРНЫ:
- Микросервисная архитектура и распределенные системы
- Event-driven architecture и CQRS
- Domain-Driven Design (DDD) и Clean Architecture
- Hexagonal Architecture и Onion Architecture
- Saga паттерн для распределенных транзакций
- Circuit Breaker и Bulkhead паттерны

ИНФРАСТРУКТУРА И DEVOPS:
- Kubernetes: deployment, services, ingress, operators
- Docker: multi-stage builds, security, optimization
- AWS: EC2, ECS, EKS, Lambda, API Gateway, RDS, DynamoDB
- GCP: GKE, Cloud Functions, Cloud SQL, BigQuery
- Azure: AKS, Functions, CosmosDB, Service Bus
- Terraform и Ansible для Infrastructure as Code
- Jenkins, GitLab CI, GitHub Actions для CI/CD
- Prometheus, Grafana, ELK Stack для мониторинга

БАЗЫ ДАННЫХ И ХРАНИЛИЩА:
- Реляционные: PostgreSQL, MySQL, SQL Server с оптимизацией запросов
- NoSQL: MongoDB, Cassandra, DynamoDB для масштабируемости
- Кеширование: Redis, Memcached, CDN стратегии
- Поисковые системы: Elasticsearch, Solr
- Временные ряды: InfluxDB, TimescaleDB

БЕЗОПАСНОСТЬ И ПРОИЗВОДИТЕЛЬНОСТЬ:
- OAuth 2.0, JWT, SAML для аутентификации
- Защита от OWASP Top 10 уязвимостей
- Оптимизация производительности и профилирование
- Масштабирование горизонтальное и вертикальное

Отвечай всегда подробно, с практическими примерами кода и архитектурными диаграммами где это уместно. Используй лучшие практики индустрии.""",
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        test_variations = [
            "Объясни принципы SOLID",
            "Что такое микросервисы?", 
            "Как работает Docker?"
        ]
        
        responses = []
        
        for i, question in enumerate(test_variations):
            messages = [
                {"role": "system", "content": long_system_message},
                {"role": "user", "content": question}
            ]
            
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.1
            }
            
            try:
                def make_request():
                    return self.openai_client.chat.completions.create(**request_data)
                
                response = self.retry_operation(make_request, f"system message test {i+1}")
                self._log_cache_response("cache_system_message", model, request_data, response, i+1)
                
                responses.append({
                    "question": question,
                    "attempt": i+1,
                    "usage": response.usage.__dict__ if response.usage else {},
                    "cache_info": self._extract_cache_info(response)
                })
                
                time.sleep(2)
                
            except Exception as e:
                responses.append({
                    "question": question,
                    "attempt": i+1,
                    "error": str(e)
                })
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "system_message_length": len(long_system_message),
                "responses": responses,
                "log_file": str(self.cache_log_file)
            }
        )
    
    def _test_cache_token_analysis(self, model: str, **kwargs) -> TestResult:
        """Детальный анализ токенов кеширования"""
        self.log("INFO", f"Анализ токенов кеширования для {model}")
        
        if not self.openai_client:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="OpenAI client not available"
            )
        
        # Создаем запрос специально для анализа токенов с cache_control
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Ты специалист по оптимизации производительности приложений с многолетним опытом работы в высоконагруженных системах. Твоя экспертиза охватывает: алгоритмы кеширования (LRU, LFU, FIFO), распределенные кеши (Redis Cluster, Memcached), CDN стратегии, оптимизацию баз данных, индексирование, партиционирование, репликацию, load balancing, horizontal и vertical scaling, мониторинг производительности, профилирование приложений. Рассматривай каждый вопрос с точки зрения производительности, масштабируемости и эффективности использования ресурсов.",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user", 
                "content": "Расскажи о стратегиях кеширования данных в высоконагруженных системах."
            }
        ]
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.0
        }
        
        token_analysis = []
        
        # Делаем несколько запросов для анализа
        for i in range(3):
            try:
                def make_request():
                    return self.openai_client.chat.completions.create(**request_data)
                
                response = self.retry_operation(make_request, f"token analysis {i+1}")
                self._log_cache_response("cache_token_analysis", model, request_data, response, i+1)
                
                usage_info = response.usage.__dict__ if response.usage else {}
                cache_info = self._extract_cache_info(response)
                
                token_analysis.append({
                    "request_number": i+1,
                    "timestamp": datetime.now().isoformat(),
                    "usage": usage_info,
                    "cache_info": cache_info,
                    "total_tokens": usage_info.get('total_tokens', 0),
                    "prompt_tokens": usage_info.get('prompt_tokens', 0),
                    "completion_tokens": usage_info.get('completion_tokens', 0),
                    "cache_creation_input_tokens": usage_info.get('cache_creation_input_tokens', 0),
                    "cache_read_input_tokens": usage_info.get('cache_read_input_tokens', 0)
                })
                
                time.sleep(3)
                
            except Exception as e:
                token_analysis.append({
                    "request_number": i+1,
                    "error": str(e)
                })
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "token_analysis": token_analysis,
                "cache_efficiency": self._calculate_cache_efficiency(token_analysis),
                "log_file": str(self.cache_log_file)
            }
        )
    
    def _compare_cache_responses(self, responses: List[Dict]) -> Dict:
        """Сравнивает ответы для поиска признаков кеширования"""
        if len(responses) < 2:
            return {"error": "Not enough responses to compare"}
        
        comparison = {
            "identical_content": all(r.get("content") == responses[0].get("content") for r in responses),
            "usage_differences": [],
            "cache_info_found": False
        }
        
        for i, resp in enumerate(responses):
            usage = resp.get("usage", {})
            cache_info = resp.get("cache_info", {})
            
            if cache_info:
                comparison["cache_info_found"] = True
            
            comparison["usage_differences"].append({
                "attempt": i + 1,
                "total_tokens": usage.get("total_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_info": cache_info
            })
        
        return comparison
    
    def _analyze_cache_patterns(self, responses: List[Dict]) -> Dict:
        """Анализирует паттерны кеширования в серии запросов"""
        analysis = {
            "total_requests": len(responses),
            "successful_requests": len([r for r in responses if "error" not in r]),
            "cache_hits_detected": 0,
            "token_savings": 0,
            "patterns": []
        }
        
        for resp in responses:
            if "error" in resp:
                continue
                
            usage = resp.get("usage", {})
            cache_info = resp.get("cache_info", {})
            
            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            if cache_read_tokens > 0:
                analysis["cache_hits_detected"] += 1
            
            pattern = {
                "attempt": resp.get("attempt", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cache_creation": usage.get("cache_creation_input_tokens", 0),
                "cache_read": cache_read_tokens,
                "has_cache_info": bool(cache_info)
            }
            
            analysis["patterns"].append(pattern)
        
        return analysis
    
    def _test_cache_support_check(self, model: str, **kwargs) -> TestResult:
        """Проверка поддержки кеширования API"""
        self.log("INFO", f"Проверка поддержки кеширования для {model}")
        
        if not self.openai_client:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="OpenAI client not available"
            )
        
        # Простой запрос для проверки поддержки
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user",
                "content": "Hello, can you tell me about caching?"
            }
        ]
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 100
        }
        
        try:
            def make_request():
                return self.openai_client.chat.completions.create(**request_data)
            
            response = self.retry_operation(make_request, "cache support check")
            self._log_cache_response("cache_support_check", model, request_data, response, 1)
            
            # Проверяем наличие информации о кешировании
            cache_info = self._extract_cache_info(response)
            usage = response.usage.__dict__ if response.usage else {}
            
            # Анализируем поддержку
            support_indicators = []
            
            if cache_info:
                support_indicators.append(f"Cache info found: {cache_info}")
            
            if any('cache' in key.lower() for key in usage.keys()):
                cache_fields = [key for key in usage.keys() if 'cache' in key.lower()]
                support_indicators.append(f"Cache fields in usage: {cache_fields}")
            
            # Проверяем наличие специфичных полей OpenRouter кеширования
            expected_cache_fields = [
                'cache_creation_input_tokens',
                'cache_read_input_tokens', 
                'cached_tokens',
                'cache_hit'
            ]
            
            found_cache_fields = []
            for field in expected_cache_fields:
                if field in usage:
                    found_cache_fields.append(field)
            
            if found_cache_fields:
                support_indicators.append(f"OpenRouter cache fields found: {found_cache_fields}")
            
            support_status = "SUPPORTED" if (cache_info or found_cache_fields) else "NOT_SUPPORTED"
            
            return TestResult(
                status=TestStatus.SUCCESS,
                data={
                    "support_status": support_status,
                    "support_indicators": support_indicators,
                    "usage": usage,
                    "cache_info": cache_info,
                    "model_tested": model,
                    "api_endpoint": self.config.base_url,
                    "log_file": str(self.cache_log_file)
                }
            )
            
        except Exception as e:
            return TestResult(
                status=TestStatus.FAILED,
                error=f"Cache support check failed: {str(e)}"
            )
    
    def _calculate_cache_efficiency(self, token_analysis: List[Dict]) -> Dict:
        """Вычисляет эффективность кеширования"""
        if not token_analysis:
            return {"error": "No data for analysis"}
        
        total_requests = len([ta for ta in token_analysis if "error" not in ta])
        cache_hits = 0
        total_cache_read = 0
        total_cache_creation = 0
        
        for analysis in token_analysis:
            if "error" in analysis:
                continue
                
            cache_read = analysis.get("cache_read_input_tokens", 0)
            cache_creation = analysis.get("cache_creation_input_tokens", 0)
            
            if cache_read > 0:
                cache_hits += 1
                total_cache_read += cache_read
            
            if cache_creation > 0:
                total_cache_creation += cache_creation
        
        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_requests if total_requests > 0 else 0,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_creation_tokens": total_cache_creation,
            "efficiency_ratio": total_cache_read / total_cache_creation if total_cache_creation > 0 else 0
        }
    
    def _test_cache_historian_example(self, model: str, **kwargs) -> TestResult:
        """Тест кеширования с примером историка, как в документации OpenRouter"""
        self.log("INFO", f"Тестирование кеширования с примером историка для {model}")
        
        if not self.openai_client:
            return TestResult(
                status=TestStatus.SKIPPED,
                error="OpenAI client not available"
            )
        
        # Пример из документации OpenRouter
        huge_text_body = """The Roman Empire's decline and fall is a complex historical phenomenon that unfolded over several centuries. The traditional date for the fall of the Western Roman Empire is 476 CE, when the Germanic chieftain Odoacer deposed the last Western Roman Emperor, Romulus Augustulus. However, this event was merely the culmination of a long process of political, economic, and military decay.

The crisis of the third century (235-284 CE) marked a turning point, characterized by political instability, economic inflation, military pressures from barbarian tribes, and the fragmentation of imperial authority. During this period, the empire was plagued by frequent changes in leadership, with over fifty emperors claiming power in fifty years, most dying violent deaths.

Economic factors played a crucial role in Rome's decline. The empire faced severe inflation due to the debasement of currency, heavy taxation to fund the military and bureaucracy, and the disruption of trade routes. The traditional Roman economy, based on slave labor and conquest, became unsustainable as expansion ceased and maintenance costs increased.

Military challenges intensified as Germanic tribes pressed against Roman frontiers, particularly along the Rhine and Danube rivers. The Huns, under Attila, displaced these tribes, creating a domino effect that increased pressure on Roman borders. The sack of Rome by Visigoth king Alaric in 410 CE and later by Vandal king Genseric in 455 CE demonstrated the empire's military vulnerability.

Administrative reforms by Diocletian (284-305 CE) temporarily stabilized the empire through the Tetrarchy system, dividing power among four rulers. However, this division eventually led to permanent separation between East and West. Constantine's founding of Constantinople in 330 CE shifted the empire's center eastward, leaving the Western provinces increasingly isolated.

Religious transformation also marked this period. Constantine's conversion to Christianity in the early 4th century fundamentally changed Roman society and politics. While Christianity provided some unity, it also created new divisions and weakened traditional Roman civic values and institutions."""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a historian studying the fall of the Roman Empire. You know the following book very well:"
                    },
                    {
                        "type": "text",
                        "text": huge_text_body,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user",
                "content": "What were the main factors that led to the fall of the Western Roman Empire?"
            }
        ]
        
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.2
        }
        
        responses = []
        
        # Делаем несколько запросов с одним и тем же кешируемым контентом
        for attempt in range(3):
            try:
                def make_request():
                    return self.openai_client.chat.completions.create(**request_data)
                
                response = self.retry_operation(make_request, f"historian example attempt {attempt+1}")
                self._log_cache_response("cache_historian_example", model, request_data, response, attempt+1)
                
                responses.append({
                    "attempt": attempt + 1,
                    "timestamp": datetime.now().isoformat(),
                    "content": response.choices[0].message.content[:150] + "...",
                    "usage": response.usage.__dict__ if response.usage else {},
                    "cache_info": self._extract_cache_info(response)
                })
                
                # Пауза между запросами
                if attempt < 2:
                    time.sleep(3)
                    
            except Exception as e:
                self.log("ERROR", f"Historian example attempt {attempt+1} failed", e)
                responses.append({
                    "attempt": attempt + 1,
                    "error": str(e)
                })
        
        self.log("SUCCESS", f"Тест с примером историка завершен для {model}")
        
        return TestResult(
            status=TestStatus.SUCCESS,
            data={
                "responses": responses,
                "cached_text_length": len(huge_text_body),
                "cache_analysis": self._analyze_cache_patterns(responses),
                "log_file": str(self.cache_log_file)
            }
        )
