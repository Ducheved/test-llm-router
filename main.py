#!/usr/bin/env python3
"""
OpenRouter API Test Suite v2.0
Модульная архитектура для комплексного тестирования всех возможностей OpenRouter API

Автор: Senior Developer with 10+ years experience
Архитектура: Модульная, расширяемая, с правильным разделением ответственности

Поддерживаемые тесты:
- Chat Completions API (/api/v1/chat/completions)
- Harmony Format для reasoning моделей (gpt-oss)
- OpenRouter-специфичные endpoints (/api/v1/generation, /api/v1/models)
- Image Generation (прямой API и через chat)
- Vision/Multimodal анализ
- Tool Calling
- Streaming
- JSON Mode
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import OpenRouterConfig
from src.runner import OpenRouterTestRunner
from rich.console import Console

console = Console()


def main():
    """Главная функция приложения"""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="OpenRouter API Test Suite v2.0 - Модульная архитектура",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --models openai/gpt-4o openai/gpt-oss-120b
  python main.py --categories chat harmony imagegen
  python main.py --models anthropic/claude-3-haiku --categories vision tools
  
Поддерживаемые категории:
  chat        - Базовые chat completions
  stream      - Streaming responses
  vision      - Анализ изображений (multimodal)
  json        - JSON mode
  harmony     - Harmony format для reasoning (gpt-oss)
  tools       - Function/Tool calling
  imagegen    - Генерация изображений
  generation  - OpenRouter generation stats API
  completions - Completions API (не chat)
  models      - Models list и endpoints
  batch       - Batch запросы
  cache       - Prompt caching тестирование

Переменные окружения:
  ROUTER_API_KEY     - API ключ OpenRouter (обязательно)
  ROUTER_BASE_URL    - Base URL (по умолчанию: https://openrouter.ai)
  TEST_MODELS        - Модели через запятую
  TEST_CATEGORIES    - Категории через запятую
  VISION_IMAGE       - Путь к изображению для vision тестов
        """
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Модели для тестирования (например: openai/gpt-4o anthropic/claude-3-haiku)"
    )
    parser.add_argument(
        "--categories", 
        nargs="+", 
        help="Категории тестов (например: chat harmony imagegen)"
    )
    parser.add_argument(
        "--api-key", 
        help="API ключ OpenRouter (или установите ROUTER_API_KEY)"
    )
    parser.add_argument(
        "--base-url", 
        help="Base URL для API (по умолчанию: https://openrouter.ai)"
    )
    parser.add_argument(
        "--output-dir",
        help="Директория для результатов (по умолчанию: out)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Подробный вывод"
    )
    parser.add_argument(
        "--version", 
        action="version",
        version="OpenRouter Test Suite v2.0"
    )
    
    args = parser.parse_args()
    
    try:
        config = create_config_from_args(args)
        show_startup_info(config, args)
        runner = OpenRouterTestRunner(config)
        
        try:
            results = runner.run_full_test_suite(config.test_models, config.test_categories)
            total_success = sum(suite.success_count for suite in results.values())
            total_tests = sum(len(suite.results) for suite in results.values())
            
            if total_tests == 0:
                console.print("[yellow]⚠️ Не было выполнено ни одного теста[/]")
                sys.exit(2)
            elif total_success == 0:
                console.print("[red]❌ Все тесты провалились[/]") 
                sys.exit(1)
            else:
                success_rate = total_success / total_tests * 100
                if success_rate >= 80:
                    console.print(f"[green]✅ Тестирование успешно завершено ({success_rate:.1f}% успеха)[/]")
                else:
                    console.print(f"[yellow]⚠️ Тестирование завершено с предупреждениями ({success_rate:.1f}% успеха)[/]")
                    sys.exit(0)  # Не критическая ошибка
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Тестирование прервано пользователем[/]")
            sys.exit(130)
        
        finally:
            runner.cleanup()
    
    except ValueError as e:
        console.print(f"[red]❌ Ошибка конфигурации: {e}[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]💥 Критическая ошибка: {e}[/]")
        if args.verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)


def create_config_from_args(args) -> OpenRouterConfig:
    """Создать конфигурацию из аргументов командной строки"""
    api_key = args.api_key or os.getenv("ROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "API ключ не найден. Установите переменную ROUTER_API_KEY "
            "или используйте --api-key"
        )
    base_url = args.base_url or os.getenv("ROUTER_BASE_URL", "https://openrouter.ai")
    models = []
    if args.models:
        models = args.models
    elif os.getenv("TEST_MODELS"):
        models = [m.strip() for m in os.getenv("TEST_MODELS").split(",") if m.strip()]
    else:
        models = [
            "openai/gpt-4o",           # Multimodal, tools
            "openai/gpt-oss-120b",     # Harmony format, reasoning
            "anthropic/claude-3-haiku", # Fast, reliable
            "google/gemini-pro"        # Alternative provider
        ]
    categories = set()
    if args.categories:
        categories = set(args.categories)
    elif os.getenv("TEST_CATEGORIES"):
        categories = {c.strip() for c in os.getenv("TEST_CATEGORIES").split(",") if c.strip()}
    else:
        categories = {
            "chat", "stream", "vision", "json", 
            "harmony", "tools", "generation", "models"
        }
    config = OpenRouterConfig(
        api_key=api_key,
        base_url=base_url,
        test_models=models,
        test_categories=categories
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(exist_ok=True)
    
    return config


def show_startup_info(config: OpenRouterConfig, args):
    """Показать информацию о запуске"""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        f"[bold cyan]🚀 OpenRouter Test Suite v2.0[/]\n\n"
        f"[green]Архитектура:[/] Модульная, расширяемая\n"
        f"[blue]Endpoint:[/] {config.base_url}\n"
        f"[yellow]Моделей для тестирования:[/] {len(config.test_models)}\n"
        f"[magenta]Категорий тестов:[/] {len(config.test_categories)}\n\n"
        f"[dim]Модели:[/] {', '.join(config.test_models[:3])}{'...' if len(config.test_models) > 3 else ''}\n"
        f"[dim]Категории:[/] {', '.join(sorted(config.test_categories))}",
        title="⚙️ Конфигурация тестирования",
        border_style="cyan"
    ))
    
    if args.verbose:
        console.print("[dim]Запущен в режиме подробного вывода[/]")


if __name__ == "__main__":
    main()

