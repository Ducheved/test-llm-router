#!/usr/bin/env python3
"""
OpenRouter Prompt Caching Test Script
Специальный скрипт для тестирования и анализа кеширования промптов
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import OpenRouterConfig
from src.modules.prompt_caching import PromptCachingTestModule
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main():
    """Запуск тестов кеширования промптов"""
    load_dotenv()
    
    console.print(Panel.fit(
        "[bold cyan]🚀 OpenRouter Prompt Caching Test Suite[/]\n\n"
        "[green]Тестируем функциональность кеширования промптов[/]\n"
        "[blue]Документация:[/] https://openrouter.ai/docs/features/prompt-caching\n"
        "[yellow]Логирование:[/] Отдельный файл для анализа кеш-токенов",
        title="⚡ Cache Testing",
        border_style="cyan"
    ))
    
    try:
        config = OpenRouterConfig.from_env()
        
        # Показываем конфигурацию
        console.print(f"\n[bold]Конфигурация:[/]")
        console.print(f"  API Base URL: {config.base_url}")
        console.print(f"  Logs Directory: {config.logs_dir}")
        
    except ValueError as e:
        console.print(f"[red]❌ Ошибка конфигурации: {e}[/]")
        return
    
    # Создаем модуль для тестирования кеширования
    try:
        import httpx
        http_client = httpx.Client(timeout=config.timeout)
    except ImportError:
        http_client = None
        console.print("[yellow]⚠️ httpx не установлен, используем только OpenAI client[/]")
    
    cache_module = PromptCachingTestModule(config, http_client)
    
    # Список тестов для выполнения
    tests_to_run = [
        ("cache_basic_test", "Базовый тест кеширования"),
        ("cache_repeated_requests", "Повторяющиеся запросы"),
        ("cache_system_message", "Кеширование системных сообщений"),
        ("cache_token_analysis", "Анализ токенов кеширования"),
        ("cache_historian_example", "Пример историка (из документации)")
    ]
    
    # Модели для тестирования
    models_to_test = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "openai/gpt-4o"
    ]
    
    results = {}
    
    for model in models_to_test:
        console.rule(f"[bold cyan]🧪 Тестирование модели: {model}[/]")
        model_results = {}
        
        for test_name, test_description in tests_to_run:
            console.print(f"\n[bold]Запуск теста:[/] {test_description}")
            
            try:
                result = cache_module.run_test(test_name, model)
                model_results[test_name] = result
                
                if result.is_success():
                    console.print(f"  ✅ {test_description} - Успешно")
                    console.print(f"  ⏱️ Время выполнения: {result.duration:.2f}с")
                    
                    # Показываем ключевую информацию о кешировании
                    if result.data and "cache_comparison" in result.data:
                        cache_info = result.data["cache_comparison"]
                        if cache_info.get("cache_info_found"):
                            console.print("  💾 Информация о кеше найдена в ответе!")
                        
                    if result.data and "cache_analysis" in result.data:
                        analysis = result.data["cache_analysis"]
                        cache_hits = analysis.get("cache_hits_detected", 0)
                        if cache_hits > 0:
                            console.print(f"  🎯 Обнаружено кеш-попаданий: {cache_hits}")
                            
                    if result.data and "cache_efficiency" in result.data:
                        efficiency = result.data["cache_efficiency"]
                        hit_rate = efficiency.get("cache_hit_rate", 0)
                        if hit_rate > 0:
                            console.print(f"  📊 Коэффициент попадания в кеш: {hit_rate:.2%}")
                        
                else:
                    console.print(f"  ❌ {test_description} - Ошибка: {result.error}")
                    
            except Exception as e:
                console.print(f"  💥 {test_description} - Исключение: {str(e)}")
                model_results[test_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        results[model] = model_results
        
        # Показываем краткую сводку по модели
        successful_tests = sum(1 for r in model_results.values() if hasattr(r, 'is_success') and r.is_success())
        console.print(f"\n[bold]Итого по модели {model}:[/] {successful_tests}/{len(tests_to_run)} тестов прошли успешно")
    
    # Создаем итоговый отчет
    console.rule("[bold green]📊 Итоговый отчет по кешированию[/]")
    
    # Таблица результатов
    table = Table(title="Результаты тестирования кеширования промптов")
    table.add_column("Модель", style="cyan")
    table.add_column("Базовый тест", justify="center")
    table.add_column("Повторные запросы", justify="center")
    table.add_column("Системные сообщения", justify="center")
    table.add_column("Анализ токенов", justify="center")
    table.add_column("Пример историка", justify="center")
    
    for model, model_results in results.items():
        row = [model]
        for test_name, _ in tests_to_run:
            result = model_results.get(test_name)
            if hasattr(result, 'is_success'):
                icon = "✅" if result.is_success() else "❌"
            else:
                icon = "💥"
            row.append(icon)
        table.add_row(*row)
    
    console.print(table)
    
    # Сохраняем подробный отчет
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path("logs") / f"cache_test_report_{timestamp}.json"
    Path("logs").mkdir(exist_ok=True)
    
    report_data = {
        "timestamp": timestamp,
        "test_type": "prompt_caching",
        "models_tested": list(results.keys()),
        "tests_executed": [test_name for test_name, _ in tests_to_run],
        "results": {}
    }
    
    for model, model_results in results.items():
        report_data["results"][model] = {}
        for test_name, result in model_results.items():
            if hasattr(result, 'to_dict'):
                report_data["results"][model][test_name] = result.to_dict()
            else:
                report_data["results"][model][test_name] = result
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n[green]📄 Подробный отчет сохранен в:[/] {report_file}")
    
    # Показываем путь к логам кеширования
    if hasattr(cache_module, 'cache_log_file'):
        console.print(f"[blue]📋 Логи кеширования сохранены в:[/] {cache_module.cache_log_file}")
    
    console.print(f"\n[bold green]🎉 Тестирование кеширования завершено![/]")
    console.print(f"[dim]Для анализа кеш-токенов изучите файлы логов выше.[/]")


if __name__ == "__main__":
    main()
