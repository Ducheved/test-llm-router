"""
OpenRouter Test Suite Modules
Модульные тестеры для различных API endpoints
"""

from .chat_completions import ChatCompletionsTestModule
from .harmony import HarmonyTestModule  
from .openrouter_api import OpenRouterAPITestModule
from .image_generation import ImageGenerationTestModule

__all__ = [
    "ChatCompletionsTestModule",
    "HarmonyTestModule", 
    "OpenRouterAPITestModule",
    "ImageGenerationTestModule"
]

