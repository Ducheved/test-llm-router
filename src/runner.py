"""
Main Test Runner - Модульная архитектура тестирования OpenRouter
"""
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import httpx
except ImportError:
    httpx = None

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from .config import OpenRouterConfig
from .core import TestSuite, TestResult, TestStatus
from .modules.chat_completions import ChatCompletionsTestModule
from .modules.harmony import HarmonyTestModule  
from .modules.openrouter_api import OpenRouterAPITestModule
from .modules.image_generation import ImageGenerationTestModule

console = Console()


class OpenRouterTestRunner:
    """Главный класс для запуска всех тестов"""
    
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        if httpx:
            self.http_client = httpx.Client(timeout=config.timeout)
        else:
            self.http_client = None
        self.modules = {
            "chat": ChatCompletionsTestModule(config, self.http_client),
            "harmony": HarmonyTestModule(config, self.http_client),
            "openrouter": OpenRouterAPITestModule(config, self.http_client),
            "imagegen": ImageGenerationTestModule(config, self.http_client)
        }
        self.category_mapping = {
            "chat": [("chat", "basic_chat"), ("chat", "streaming_chat"), ("chat", "json_mode")],
            "stream": [("chat", "streaming_chat")],
            "vision": [("chat", "vision_analysis")],
            "json": [("chat", "json_mode")],
            "harmony": [("harmony", "reasoning_structured"), ("harmony", "high_reasoning")],
            "tools": [("chat", "tool_calling")],
            "imagegen": [("imagegen", "direct_image_generation"), ("imagegen", "chat_based_generation")],
            "generation": [("openrouter", "generation_stats")],
            "completions": [("openrouter", "completions_api")],
            "models": [("openrouter", "models_list"), ("openrouter", "model_endpoints")],
            "batch": [("chat", "basic_chat")],
        }
        self.results = {}
        self.start_time = datetime.now()
    
    def run_full_test_suite(self, models: List[str], categories: Set[str]) -> Dict[str, TestSuite]:
        """Запустить полный набор тестов"""
        console.print(Panel.fit(
            f"[bold cyan]🚀 OpenRouter Test Suite v2.0[/]\n\n"
            f"[green]Архитектура:[/] Модульная\n"
            f"[blue]Моделей:[/] {len(models)}\n" 
            f"[yellow]Категорий:[/] {len(categories)}\n"
            f"[magenta]Всего тестов:[/] {self._count_total_tests(categories)}",
            title="⚙️ Запуск тестирования",
            border_style="cyan"
        ))
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for model in models:
                console.rule(f"[bold cyan]🧪 Тестирование модели: {model}[/]")
                
                model_suite = TestSuite(
                    name=model,
                    results={},
                    started_at=datetime.now()
                )
                tests_to_run = self._get_tests_for_categories(categories)
                
                task = progress.add_task(
                    f"Тестирование {model}...", 
                    total=len(tests_to_run)
                )
                
                for module_name, test_name in tests_to_run:
                    module = self.modules[module_name]
                    
                    try:
                        result = module.run_test(test_name, model)
                        test_key = f"{module_name}:{test_name}"
                        model_suite.results[test_key] = result
                        status_icon = {
                            TestStatus.SUCCESS: "✅",
                            TestStatus.FAILED: "❌", 
                            TestStatus.SKIPPED: "⏭️",
                            TestStatus.ERROR: "💥"
                        }[result.status]
                        
                        console.print(f"  {status_icon} {test_key}")
                        
                    except Exception as e:
                        error_result = TestResult(
                            status=TestStatus.ERROR,
                            error=f"Unexpected error: {str(e)}"
                        )
                        test_key = f"{module_name}:{test_name}"
                        model_suite.results[test_key] = error_result
                        console.print(f"  💥 {test_key} - Unexpected error")
                    
                    progress.advance(task)
                
                model_suite.completed_at = datetime.now()
                results[model] = model_suite
        self._generate_report(results, categories)
        
        return results
    
    def _get_tests_for_categories(self, categories: Set[str]) -> List[tuple]:
        """Получить список тестов для выбранных категорий"""
        tests = []
        for category in categories:
            if category in self.category_mapping:
                tests.extend(self.category_mapping[category])
            else:
                console.print(f"[yellow]⚠️ Неизвестная категория: {category}[/]")
        return list(set(tests))
    
    def _count_total_tests(self, categories: Set[str]) -> int:
        """Подсчитать общее количество тестов"""
        tests = self._get_tests_for_categories(categories)
        return len(tests)
    
    def _generate_report(self, results: Dict[str, TestSuite], categories: Set[str]):
        """Сгенерировать итоговый отчет"""
        table = Table(title="📊 Результаты тестирования OpenRouter API", box=box.ROUNDED)
        table.add_column("Модель", style="cyan", width=40)
        all_tests = set()
        for suite in results.values():
            all_tests.update(suite.results.keys())
        all_tests = sorted(all_tests)
        
        for test in all_tests:
            table.add_column(test.replace(":", "\\n"), justify="center", width=8)
        for model_name, suite in results.items():
            row = [model_name]
            
            for test in all_tests:
                if test in suite.results:
                    result = suite.results[test]
                    icon = {
                        TestStatus.SUCCESS: "✅",
                        TestStatus.FAILED: "❌",
                        TestStatus.SKIPPED: "⏭️", 
                        TestStatus.ERROR: "💥"
                    }[result.status]
                    row.append(icon)
                else:
                    row.append("➖")
            
            table.add_row(*row)
        
        console.print("\\n")
        console.print(table)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.config.output_dir / f"test_report_v2_{timestamp}.json"
        json_data = {}
        for model_name, suite in results.items():
            json_data[model_name] = {
                "started_at": suite.started_at.isoformat(),
                "completed_at": suite.completed_at.isoformat() if suite.completed_at else None,
                "duration": suite.duration,
                "success_rate": suite.success_rate,
                "results": {k: v.to_dict() for k, v in suite.results.items()}
            }
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "version": "2.0",
                    "timestamp": timestamp,
                    "categories_tested": list(categories),
                    "total_models": len(results),
                    "architecture": "modular"
                },
                "results": json_data
            }, f, ensure_ascii=False, indent=2)
        total_tests = sum(len(suite.results) for suite in results.values())
        successful_tests = sum(suite.success_count for suite in results.values())
        overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        console.print(Panel.fit(
            f"[green bold]🎉 Тестирование завершено![/]\\n\\n"
            f"[blue]📈 Статистика:[/]\\n"
            f"• Моделей протестировано: [bold]{len(results)}[/]\\n"
            f"• Всего тестов выполнено: [bold]{total_tests}[/]\\n" 
            f"• Успешных тестов: [bold green]{successful_tests}[/]\\n"
            f"• Общий процент успеха: [bold]{overall_success_rate:.1f}%[/]\\n\\n"
            f"[yellow]📁 Файлы:[/]\\n"
            f"• Детальный отчет: [blue]{report_file}[/]\\n"
            f"• Изображения: [blue]{self.config.output_dir}[/]",
            title="✨ Итоговый отчет",
            border_style="green"
        ))
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.http_client and hasattr(self.http_client, 'close'):
            self.http_client.close()


def main():
    """Главная функция"""
    try:
        config = OpenRouterConfig.from_env()
        models = config.test_models or [
            "openai/gpt-4o", 
            "openai/gpt-oss-120b",
            "google/gemini-pro",
            "anthropic/claude-3-haiku"
        ]
        
        categories = config.test_categories or {
            "chat", "stream", "vision", "json", 
            "harmony", "tools", "generation", "models"
        }
        
        runner = OpenRouterTestRunner(config)
        
        try:
            results = runner.run_full_test_suite(models, categories)
            total_success = sum(suite.success_count for suite in results.values())
            total_tests = sum(len(suite.results) for suite in results.values())
            
            if total_success == 0 and total_tests > 0:
                sys.exit(1)
            
        except KeyboardInterrupt:
            console.print("\\n[yellow]⚠️ Тестирование прервано пользователем[/]")
            sys.exit(130)
        
        finally:
            runner.cleanup()
    
    except Exception as e:
        console.print(f"\\n[red]💥 Критическая ошибка: {e}[/]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()

