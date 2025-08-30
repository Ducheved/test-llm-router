"""
OpenRouter API Test Suite - Core Configuration
Модульная архитектура для тестирования всех возможностей OpenRouter
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path

@dataclass
class OpenRouterConfig:
    """Централизованная конфигурация для всех тестов"""
    api_key: str
    base_url: str = "https://openrouter.ai"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: int = 2
    test_models: List[str] = field(default_factory=list)
    test_categories: Set[str] = field(default_factory=set)
    output_dir: Path = field(default_factory=lambda: Path("out"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    payloads_dir: Path = field(default_factory=lambda: Path("payloads"))
    vision_image_path: str = "payloads/cat.jpeg"
    multimodal_keywords: Set[str] = field(default_factory=lambda: {
        "vision", "4o", "4-vision", "claude-3", "gemini", "qwen", "gpt-5",
        "llama-vision", "pixtral", "llava"
    })
    
    reasoning_models: Set[str] = field(default_factory=lambda: {
        "gpt-oss", "o1", "o3", "reasoning"
    })
    
    image_gen_keywords: Set[str] = field(default_factory=lambda: {
        "dall", "flux", "stable", "midjourney", "imagen"
    })
    
    @classmethod
    def from_env(cls) -> 'OpenRouterConfig':
        """Создать конфигурацию из переменных окружения"""
        api_key = os.getenv("ROUTER_API_KEY")
        if not api_key:
            raise ValueError("ROUTER_API_KEY не найден в переменных окружения")
        base_url = os.getenv("ROUTER_BASE_URL", "https://openrouter.ai")
        
        test_models = []
        models_env = os.getenv("TEST_MODELS", "")
        if models_env:
            test_models = [m.strip() for m in models_env.split(",") if m.strip()]
        
        test_categories = set()
        categories_env = os.getenv("TEST_CATEGORIES", "")
        if categories_env:
            test_categories = {c.strip() for c in categories_env.split(",") if c.strip()}
        else:
            test_categories = {
                "chat", "stream", "vision", "json", "harmony", 
                "tools", "batch", "generation", "completions", "models", "imagegen"
            }
        
        config = cls(
            api_key=api_key,
            base_url=base_url,
            timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY", "2")),
            test_models=test_models,
            test_categories=test_categories,
            vision_image_path=os.getenv("VISION_IMAGE", "payloads/cat.jpeg")
        )
        config.output_dir.mkdir(exist_ok=True)
        config.logs_dir.mkdir(exist_ok=True)
        config.payloads_dir.mkdir(exist_ok=True)
        
        return config

    def get_api_url(self, endpoint: str) -> str:
        """Построить полный URL для API endpoint"""
        if endpoint:
            return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        else:
            return self.base_url.rstrip('/')

