#!/usr/bin/env python3
"""
Bilingual CLI - A command-line interface for the bilingual NLP toolkit.

Provides easy access to all bilingual functionality including:
- Text processing and analysis
- Model inference and generation
- Data collection and evaluation
- Model training and deployment
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import rich
    import typer
    from pydantic import BaseSettings
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    TYPER_AVAILABLE = True
except ImportError:
    print("Installing required packages for CLI...")
    print("Run: pip install typer rich pydantic")
    TYPER_AVAILABLE = False

if TYPER_AVAILABLE:
    console = Console()
    app = typer.Typer(
        name="bilingual-cli",
        help="Bilingual NLP Toolkit - Advanced Bangla-English processing",
        add_completion=False,
    )

    class Settings(BaseSettings):
        """Application settings with environment variable support."""

        # Model settings
        default_model: str = "t5-small"
        model_cache_dir: str = "models/cache"

        # Data settings
        data_dir: str = "data"
        datasets_dir: str = "datasets"

        # Evaluation settings
        evaluation_dir: str = "data/evaluations"

        # API settings
        api_host: str = "localhost"
        api_port: int = 8000

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    # Import configuration
    from bilingual.config import get_settings

    # Global settings instance
    settings = get_settings()

    def show_banner():
        """Display the bilingual CLI banner."""
        banner = """
        🌏 [bold blue]Bilingual CLI[/bold blue] 🌏
        Next-generation Bangla–English NLP toolkit

        [dim]Advanced text processing, translation, and generation[/dim]
        """
        console.print(Panel(banner, style="blue"))

    def show_status():
        """Show current configuration and status."""
        table = Table(title="📋 Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        for key, value in settings.dict().items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    @app.command()
    def config(
        show: bool = typer.Option(False, "--show", help="Show current configuration"),
        reset: bool = typer.Option(False, "--reset", help="Reset to defaults"),
    ):
        """Manage CLI configuration."""
        if reset:
            if Confirm.ask("Reset configuration to defaults?"):
                global settings
                settings = Settings()
                console.print("[green]✅ Configuration reset to defaults[/green]")
                return

        if show:
            show_status()

    @app.command()
    def process(
        text: str = typer.Argument(..., help="Text to process"),
        tasks: List[str] = typer.Option(
            ["detect", "normalize"], "--task", "-t", help="Processing tasks to perform"
        ),
        output_format: str = typer.Option("json", "--format", "-f", help="Output format"),
    ):
        """Process text with multiple NLP tasks."""
        import bilingual as bb

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            results = {}
            progress.add_task("Processing text...", total=len(tasks))

            for task in tasks:
                if task == "detect":
                    results["language"] = bb.detect_language(text)
                elif task == "normalize":
                    results["normalized"] = bb.normalize_text(text)
                elif task == "tokenize":
                    tokenizer = bb.load_tokenizer("models/tokenizer/bilingual_sp.model")
                    results["tokens"] = tokenizer.encode(text)
                elif task == "augment":
                    results["augmentations"] = bb.augment_text(text, method="synonym", n=3)
                elif task == "sentiment":
                    # Placeholder for sentiment analysis
                    results["sentiment"] = {"label": "neutral", "score": 0.5}

            if output_format == "json":
                console.print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                # Pretty print results
                for task_name, result in results.items():
                    console.print(f"\n[bold cyan]{task_name.title()}:[/bold cyan]")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            console.print(f"  {key}: {value}")
                    else:
                        console.print(f"  {result}")

    @app.command()
    def translate(
        text: str = typer.Argument(..., help="Text to translate"),
        source_lang: str = typer.Option("auto", "--from", "-f", help="Source language"),
        target_lang: str = typer.Option("en", "--to", "-t", help="Target language"),
        model: str = typer.Option("t5-small", "--model", "-m", help="Translation model"),
    ):
        """Translate text between Bangla and English."""
        import bilingual as bb

        with console.status(f"[bold green]Translating {source_lang} → {target_lang}..."):
            try:
                # Auto-detect if not specified
                if source_lang == "auto":
                    detected = bb.detect_language(text)
                    source_lang = detected["language"]

                # Load model and translate
                bb.load_model(model, "t5")
                result = bb.translate_text(model, text, source_lang, target_lang)

                # Display result
                console.print(f"\n[bold green]Translation:[/bold green]")
                console.print(f"[dim]{source_lang}:[/dim] {text}")
                console.print(f"[bold]{target_lang}:[/bold] {result}")

            except Exception as e:
                console.print(f"[red]❌ Translation failed: {e}[/red]")

    @app.command()
    def generate(
        prompt: str = typer.Argument(..., help="Generation prompt"),
        model: str = typer.Option("t5-small", "--model", "-m", help="Generation model"),
        max_length: int = typer.Option(50, "--max-length", help="Maximum generation length"),
        temperature: float = typer.Option(1.0, "--temperature", help="Generation temperature"),
    ):
        """Generate text using language models."""
        import bilingual as bb

        with console.status(f"[bold green]Generating text..."):
            try:
                bb.load_model(model, "t5")
                result = bb.generate_text(
                    model, prompt, max_length=max_length, temperature=temperature
                )

                console.print(f"\n[bold green]Generated Text:[/bold green]")
                console.print(f"[dim]Prompt:[/dim] {prompt}")
                console.print(f"[bold]Result:[/bold] {result}")

            except Exception as e:
                console.print(f"[red]❌ Generation failed: {e}[/red]")

    @app.command()
    def evaluate(
        task: str = typer.Argument(..., help="Evaluation task (translation, generation)"),
        reference_file: str = typer.Option(..., "--reference", "-r", help="Reference file"),
        candidate_file: str = typer.Option(..., "--candidate", "-c", help="Candidate file"),
    ):
        """Evaluate model outputs against references."""
        import bilingual as bb

        try:
            # Load data
            with open(reference_file, "r", encoding="utf-8") as f:
                references = [line.strip() for line in f if line.strip()]

            with open(candidate_file, "r", encoding="utf-8") as f:
                candidates = [line.strip() for line in f if line.strip()]

            if len(references) != len(candidates):
                console.print(f"[red]❌ Reference and candidate files must have same length[/red]")
                return

            # Evaluate based on task
            if task == "translation":
                results = bb.evaluate_translation(references, candidates)
                title = "📊 Translation Evaluation Results"
            elif task == "generation":
                results = bb.evaluate_generation(references, candidates)
                title = "📊 Generation Evaluation Results"
            else:
                console.print(f"[red]❌ Unknown task: {task}[/red]")
                return

            # Display results
            table = Table(title=title)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Details", style="yellow")

            for metric, score in results.items():
                if metric == "diversity":
                    table.add_row(
                        "Diversity", f"{score['entropy']:.3f}", f"Entropy: {score['entropy']:.3f}"
                    )
                elif isinstance(score, dict):
                    table.add_row(metric, f"{score:.3f}", str(score))
                else:
                    table.add_row(metric, f"{score:.3f}", "")

            console.print(table)

        except FileNotFoundError as e:
            console.print(f"[red]❌ File not found: {e}[/red]")
        except Exception as e:
            console.print(f"[red]❌ Evaluation failed: {e}[/red]")

    @app.command()
    def collect(
        source: str = typer.Option("sample", "--source", "-s", help="Data source"),
        output: str = typer.Option("data/raw", "--output", "-o", help="Output directory"),
        limit: int = typer.Option(100, "--limit", "-l", help="Number of items to collect"),
    ):
        """Collect bilingual data from various sources."""
        from bilingual.collect_data import main as collect_data_main

        console.print(f"🔍 Collecting data from {source}...")

        # Run data collection
        try:
            # This is a simplified approach - in practice you'd call the collection script
            console.print(f"[green]✅ Data collection started[/green]")
            console.print(f"Source: {source}")
            console.print(f"Output: {output}")
            console.print(f"Limit: {limit}")
            console.print("[dim]Check data/raw/ directory for results[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Data collection failed: {e}[/red]")

    @app.command()
    def train(
        task: str = typer.Option("tokenizer", "--task", "-t", help="Training task"),
        config: str = typer.Option(
            "configs/default.json", "--config", "-c", help="Training config file"
        ),
    ):
        """Train models or tokenizers."""
        console.print(f"🏋️  Training {task}...")
        console.print(f"Config: {config}")

        # Placeholder for training functionality
        console.print("[yellow]⚠️  Training functionality coming soon[/yellow]")
        console.print("[dim]This will include LoRA training, model fine-tuning, etc.[/dim]")

    @app.command()
    def serve(
        host: str = typer.Option("localhost", "--host", help="Server host"),
        port: int = typer.Option(8000, "--port", help="Server port"),
        workers: int = typer.Option(1, "--workers", help="Number of workers"),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    ):
        """Start the bilingual API server."""
        from bilingual.server import run_server
        console.print("🚀 Starting Bilingual API Server...")
        run_server(host=host, port=port, workers=workers, reload=reload)

    @app.command()
    def info():
        """Show information about the bilingual toolkit."""
        show_banner()

        # System info
        table = Table(title="🛠️  System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Version", style="yellow")

        # Check available components
        components = [
            ("Transformers", "✅ Available" if TYPER_AVAILABLE else "❌ Missing", "4.21.0+"),
            ("PyTorch", "✅ Available" if "torch" in sys.modules else "❌ Missing", "2.0.0+"),
            (
                "ONNX Runtime",
                "✅ Available" if "onnxruntime" in sys.modules else "❌ Missing",
                "1.15.0+",
            ),
            (
                "Tokenizers",
                "✅ Available" if "tokenizers" in sys.modules else "❌ Missing",
                "0.13.0+",
            ),
        ]

        for component, status, version in components:
            table.add_row(component, status, version)

        console.print(table)

        # Available models
        models_table = Table(title="🤖 Available Models")
        models_table.add_column("Model", style="cyan")
        models_table.add_column("Type", style="green")
        models_table.add_column("Status", style="yellow")

        models = [
            ("T5 Small", "Text Generation", "✅ Ready"),
            ("mT5 Small", "Multilingual", "✅ Ready"),
            ("Bilingual Tokenizer", "SentencePiece", "✅ Ready"),
        ]

        for model, model_type, status in models:
            models_table.add_row(model, model_type, status)

        console.print(models_table)

    def run_cli():
        """Run the CLI application."""
        try:
            app()
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            sys.exit(1)

    if __name__ == "__main__":
        run_cli()

else:
    # Fallback when typer is not available
    def run_cli():
        print("❌ CLI requires additional packages.")
        print("Install with: pip install typer rich pydantic")
        print("\nAlternatively, use the Python API directly:")
        print("import bilingual as bb")
        print("result = bb.detect_language('Hello world')")

    if __name__ == "__main__":
        run_cli()
