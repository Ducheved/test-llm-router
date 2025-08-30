#!/usr/bin/env python3
"""
Анализатор логов кеширования промптов OpenRouter
Утилита для анализа и визуализации данных кеширования
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def load_cache_logs(log_file_path: str) -> List[Dict]:
    """Загрузить логи кеширования из файла"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Разделяем записи по разделителю
        entries = content.split('---\n')
        logs = []
        
        for entry in entries:
            entry = entry.strip()
            if entry:
                try:
                    log_data = json.loads(entry)
                    logs.append(log_data)
                except json.JSONDecodeError:
                    continue
        
        return logs
    except Exception as e:
        console.print(f"[red]Ошибка загрузки логов: {e}[/]")
        return []


def analyze_cache_usage(logs: List[Dict]) -> Dict:
    """Анализ использования кеша из логов"""
    analysis = {
        "total_requests": len(logs),
        "models_tested": set(),
        "tests_executed": set(),
        "cache_stats": {
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
            "requests_with_cache_info": 0
        },
        "cache_efficiency": {},
        "token_breakdown": []
    }
    
    for log in logs:
        # Собираем статистику
        model = log.get("model", "unknown")
        test_name = log.get("test_name", "unknown")
        usage = log.get("usage", {})
        cache_info = log.get("cache_info", {})
        
        analysis["models_tested"].add(model)
        analysis["tests_executed"].add(test_name)
        
        # Анализируем токены
        total_tokens = usage.get("total_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        
        analysis["cache_stats"]["total_tokens"] += total_tokens
        analysis["cache_stats"]["cache_creation_tokens"] += cache_creation
        analysis["cache_stats"]["cache_read_tokens"] += cache_read
        
        if cache_info:
            analysis["cache_stats"]["requests_with_cache_info"] += 1
        
        # Детальная разбивка по токенам
        token_entry = {
            "timestamp": log.get("timestamp"),
            "model": model,
            "test": test_name,
            "attempt": log.get("attempt", 1),
            "total_tokens": total_tokens,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cache_creation": cache_creation,
            "cache_read": cache_read,
            "has_cache_info": bool(cache_info),
            "cache_fields": list(cache_info.keys()) if cache_info else []
        }
        
        analysis["token_breakdown"].append(token_entry)
    
    # Конвертируем sets в lists для JSON сериализации
    analysis["models_tested"] = list(analysis["models_tested"])
    analysis["tests_executed"] = list(analysis["tests_executed"])
    
    # Рассчитываем эффективность кеширования
    cache_stats = analysis["cache_stats"]
    if cache_stats["cache_creation_tokens"] > 0:
        analysis["cache_efficiency"]["efficiency_ratio"] = (
            cache_stats["cache_read_tokens"] / cache_stats["cache_creation_tokens"]
        )
    else:
        analysis["cache_efficiency"]["efficiency_ratio"] = 0
    
    analysis["cache_efficiency"]["cache_hit_percentage"] = (
        cache_stats["requests_with_cache_info"] / analysis["total_requests"] * 100
        if analysis["total_requests"] > 0 else 0
    )
    
    return analysis


def display_analysis(analysis: Dict):
    """Отображение результатов анализа"""
    console.print(Panel.fit(
        f"[bold cyan]📊 Анализ кеширования промптов OpenRouter[/]\n\n"
        f"[green]Всего запросов:[/] {analysis['total_requests']}\n"
        f"[blue]Моделей протестировано:[/] {len(analysis['models_tested'])}\n"
        f"[yellow]Тестов выполнено:[/] {len(analysis['tests_executed'])}",
        title="🔍 Общая статистика",
        border_style="cyan"
    ))
    
    # Статистика токенов
    cache_stats = analysis["cache_stats"]
    
    stats_table = Table(title="Статистика токенов кеширования", box=box.ROUNDED)
    stats_table.add_column("Метрика", style="cyan")
    stats_table.add_column("Значение", style="green", justify="right")
    stats_table.add_column("Процент", style="yellow", justify="right")
    
    total_tokens = cache_stats["total_tokens"]
    cache_creation = cache_stats["cache_creation_tokens"]
    cache_read = cache_stats["cache_read_tokens"]
    
    stats_table.add_row(
        "Общее количество токенов",
        f"{total_tokens:,}",
        "100.0%"
    )
    
    stats_table.add_row(
        "Токены создания кеша",
        f"{cache_creation:,}",
        f"{(cache_creation/total_tokens*100):.2f}%" if total_tokens > 0 else "0.00%"
    )
    
    stats_table.add_row(
        "Токены чтения кеша",
        f"{cache_read:,}",
        f"{(cache_read/total_tokens*100):.2f}%" if total_tokens > 0 else "0.00%"
    )
    
    stats_table.add_row(
        "Запросы с кеш-информацией",
        f"{cache_stats['requests_with_cache_info']}",
        f"{analysis['cache_efficiency']['cache_hit_percentage']:.1f}%"
    )
    
    console.print(stats_table)
    
    # Эффективность кеширования
    efficiency_table = Table(title="Эффективность кеширования", box=box.ROUNDED)
    efficiency_table.add_column("Метрика", style="cyan")
    efficiency_table.add_column("Значение", style="green")
    
    efficiency = analysis["cache_efficiency"]
    
    efficiency_table.add_row(
        "Коэффициент эффективности",
        f"{efficiency['efficiency_ratio']:.2f}"
    )
    
    efficiency_table.add_row(
        "Процент попаданий в кеш",
        f"{efficiency['cache_hit_percentage']:.1f}%"
    )
    
    if cache_creation > 0 and cache_read > 0:
        savings = cache_read - cache_creation
        efficiency_table.add_row(
            "Экономия токенов",
            f"{savings:,} токенов" if savings > 0 else f"{abs(savings):,} токенов (перерасход)"
        )
    
    console.print(efficiency_table)
    
    # Разбивка по моделям
    model_breakdown = {}
    for entry in analysis["token_breakdown"]:
        model = entry["model"]
        if model not in model_breakdown:
            model_breakdown[model] = {
                "requests": 0,
                "total_tokens": 0,
                "cache_creation": 0,
                "cache_read": 0,
                "cache_info_requests": 0
            }
        
        stats = model_breakdown[model]
        stats["requests"] += 1
        stats["total_tokens"] += entry["total_tokens"]
        stats["cache_creation"] += entry["cache_creation"]
        stats["cache_read"] += entry["cache_read"]
        if entry["has_cache_info"]:
            stats["cache_info_requests"] += 1
    
    model_table = Table(title="Статистика по моделям", box=box.ROUNDED)
    model_table.add_column("Модель", style="cyan")
    model_table.add_column("Запросов", justify="center")
    model_table.add_column("Всего токенов", justify="right")
    model_table.add_column("Создание кеша", justify="right")
    model_table.add_column("Чтение кеша", justify="right")
    model_table.add_column("% с кеш-инфо", justify="right")
    
    for model, stats in model_breakdown.items():
        cache_info_pct = (stats["cache_info_requests"] / stats["requests"] * 100) if stats["requests"] > 0 else 0
        
        model_table.add_row(
            model,
            str(stats["requests"]),
            f"{stats['total_tokens']:,}",
            f"{stats['cache_creation']:,}",
            f"{stats['cache_read']:,}",
            f"{cache_info_pct:.1f}%"
        )
    
    console.print(model_table)


def main():
    """Главная функция анализатора"""
    if len(sys.argv) < 2:
        console.print("[red]Использование: python analyze_cache.py <путь_к_логу_кеширования>[/]")
        console.print("[yellow]Пример: python analyze_cache.py logs/prompt_cache_20240830_120000.log[/]")
        
        # Показываем доступные лог файлы
        logs_dir = Path("logs")
        if logs_dir.exists():
            cache_logs = list(logs_dir.glob("prompt_cache_*.log"))
            if cache_logs:
                console.print(f"\n[blue]Доступные файлы логов кеширования:[/]")
                for log_file in sorted(cache_logs):
                    console.print(f"  {log_file}")
        return
    
    log_file_path = sys.argv[1]
    
    if not Path(log_file_path).exists():
        console.print(f"[red]Файл не найден: {log_file_path}[/]")
        return
    
    console.print(f"[green]Загрузка логов из:[/] {log_file_path}")
    
    logs = load_cache_logs(log_file_path)
    
    if not logs:
        console.print("[red]Не удалось загрузить логи или файл пустой[/]")
        return
    
    console.print(f"[blue]Загружено записей:[/] {len(logs)}")
    
    # Анализируем логи
    analysis = analyze_cache_usage(logs)
    
    # Отображаем результаты
    display_analysis(analysis)
    
    # Сохраняем анализ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = Path("logs") / f"cache_analysis_{timestamp}.json"
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n[green]📄 Результаты анализа сохранены в:[/] {analysis_file}")


if __name__ == "__main__":
    main()
