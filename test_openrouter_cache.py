#!/usr/bin/env python3
"""
OpenRouter Real API Cache Test
Тестирование кеширования с использованием реального OpenRouter API
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import OpenRouterConfig
from src.modules.prompt_caching import PromptCachingTestModule
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    """Тест кеширования с реальным OpenRouter API"""
    
    console.print(Panel.fit(
        "[bold red]🚨 ВАЖНО: Этот тест использует РЕАЛЬНЫЙ OpenRouter API[/]\n\n"
        "[yellow]Убедитесь что:[/]\n"
        "[green]✓[/] У вас есть действующий API ключ OpenRouter\n"
        "[green]✓[/] В .env указан правильный ROUTER_API_KEY\n"
        "[green]✓[/] ROUTER_BASE_URL=https://openrouter.ai/api/v1[/]\n\n"
        "[red]⚠️ Этот тест будет тратить реальные токены![/]",
        title="⚡ OpenRouter Cache Test",
        border_style="red"
    ))
    
    # Принудительно устанавливаем OpenRouter API
    os.environ["ROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"
    
    # Проверяем API ключ
    api_key = os.getenv("ROUTER_API_KEY")
    if not api_key or not api_key.startswith("sk-or-"):
        console.print("[red]❌ Не найден действующий OpenRouter API ключ![/]")
        console.print("[yellow]Убедитесь что ROUTER_API_KEY в .env начинается с 'sk-or-'[/]")
        return
    
    try:
        # Создаем конфигурацию с принудительным OpenRouter URL
        config = OpenRouterConfig.from_env()
        config.base_url = "https://openrouter.ai/api/v1"  # Принудительно
        
        console.print(f"\n[green]✓ Используем API:[/] {config.base_url}")
        console.print(f"[green]✓ API Key:[/] {api_key[:10]}...{api_key[-4:]}")
        
    except ValueError as e:
        console.print(f"[red]❌ Ошибка конфигурации: {e}[/]")
        return
    
    # Создаем модуль для тестирования
    try:
        import httpx
        http_client = httpx.Client(timeout=config.timeout)
    except ImportError:
        http_client = None
    
    cache_module = PromptCachingTestModule(config, http_client)
    
    # Модели поддерживающие кеширование (согласно OpenRouter docs)
    models_with_cache_support = [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet", 
        "anthropic/claude-3-opus",
        "anthropic/claude-3-5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4o-mini"
    ]
    
    console.print(f"\n[bold]Тестируем модели с поддержкой кеширования:[/]")
    for model in models_with_cache_support:
        console.print(f"  • {model}")
    
    input("\nНажмите Enter чтобы продолжить с РЕАЛЬНЫМИ токенами...")
    
    results = {}
    
    for model in models_with_cache_support:
        console.rule(f"[bold cyan]🧪 Тестируем: {model}[/]")
        
        # Сначала проверяем поддержку кеширования
        console.print("1️⃣ Проверка поддержки кеширования...")
        try:
            support_result = cache_module.run_test("cache_support_check", model)
            
            if support_result.is_success():
                support_status = support_result.data.get("support_status", "UNKNOWN")
                indicators = support_result.data.get("support_indicators", [])
                
                if support_status == "SUPPORTED":
                    console.print("[green]✅ Кеширование поддерживается![/]")
                    for indicator in indicators:
                        console.print(f"  💾 {indicator}")
                else:
                    console.print("[yellow]⚠️ Поддержка кеширования не обнаружена[/]")
                    
            else:
                console.print(f"[red]❌ Ошибка проверки: {support_result.error}[/]")
                continue
                
        except Exception as e:
            console.print(f"[red]❌ Исключение при проверке: {e}[/]")
            continue
        
        # Если поддержка найдена, делаем базовый тест
        console.print("2️⃣ Базовый тест кеширования...")
        try:
            cache_result = cache_module.run_test("cache_basic_test", model)
            
            if cache_result.is_success():
                console.print("[green]✅ Базовый тест прошел успешно[/]")
                
                # Анализируем результаты
                if "cache_comparison" in cache_result.data:
                    comparison = cache_result.data["cache_comparison"]
                    if comparison.get("cache_info_found"):
                        console.print("  🎯 Обнаружена информация о кеше!")
                        
                        usage_diffs = comparison.get("usage_differences", [])
                        for diff in usage_diffs:
                            attempt = diff.get("attempt", "?")
                            cache_creation = diff.get("cache_creation_tokens", 0)
                            cache_read = diff.get("cache_read_tokens", 0)
                            
                            console.print(f"    Попытка {attempt}: создание={cache_creation}, чтение={cache_read}")
                    else:
                        console.print("  ℹ️ Информация о кеше не найдена в ответах")
                        
            else:
                console.print(f"[red]❌ Базовый тест не прошел: {cache_result.error}[/]")
                
        except Exception as e:
            console.print(f"[red]❌ Исключение в базовом тесте: {e}[/]")
        
        results[model] = {
            "support_check": support_result.data if support_result.is_success() else {"error": support_result.error},
            "basic_test": cache_result.data if 'cache_result' in locals() and cache_result.is_success() else {"error": "Test failed"}
        }
        
        console.print()  # Пустая строка между моделями
    
    # Итоги
    console.rule("[bold green]📊 Итоги тестирования[/]")
    
    supported_models = []
    unsupported_models = []
    
    for model, result in results.items():
        support_status = result.get("support_check", {}).get("support_status", "UNKNOWN")
        if support_status == "SUPPORTED":
            supported_models.append(model)
        else:
            unsupported_models.append(model)
    
    console.print(f"[green]✅ Модели с поддержкой кеширования ({len(supported_models)}):[/]")
    for model in supported_models:
        console.print(f"  • {model}")
    
    if unsupported_models:
        console.print(f"\n[yellow]⚠️ Модели без поддержки кеширования ({len(unsupported_models)}):[/]")
        for model in unsupported_models:
            console.print(f"  • {model}")
    
    # Показываем путь к логам
    if hasattr(cache_module, 'cache_log_file'):
        console.print(f"\n[blue]📋 Подробные логи:[/] {cache_module.cache_log_file}")
        console.print("[dim]Используйте analyze_cache.py для анализа логов[/]")


if __name__ == "__main__":
    main()
