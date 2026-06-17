"""CLI entry point — calls the SDK only; no direct service imports.

Commands
--------
airllm-bench run --backend <name>
    Run one backend and print a summary.

airllm-bench run-all
    Run all local backends sequentially.

airllm-bench report
    Print a comparison table from saved results.

airllm-bench host-spec
    Print current host hardware spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from airllm_bench.constants import BackendName
from airllm_bench.sdk.analytics import (
    plot_comparison,
    render_comparison_markdown,
    summarize,
    write_comparison_table,
)
from airllm_bench.sdk.runner import BenchmarkRunner
from airllm_bench.services.host_spec import capture_host_spec, write_host_spec
from airllm_bench.shared.config import get_settings

app = typer.Typer(
    name="airllm-bench",
    help="Benchmark harness proving AirLLM value on memory-constrained hardware.",
    add_completion=False,
)


@app.command("run")
def cmd_run(
    backend: BackendName = typer.Option(..., "--backend", "-b", help="Backend to benchmark."),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Override MODEL_ID."),  # noqa: UP007
) -> None:
    """Run a single backend and print the result summary."""
    settings = get_settings()
    if model_id:
        settings = settings.model_copy(update={"model_id": model_id})
    runner = BenchmarkRunner(settings)
    result = runner.run(backend)
    _print_result(result)


@app.command("run-all")
def cmd_run_all() -> None:
    """Run all local backends (ollama, transformers-cpu, airllm) sequentially."""
    settings = get_settings()
    runner = BenchmarkRunner(settings)
    backends = [BackendName.OLLAMA, BackendName.TRANSFORMERS_CPU, BackendName.AIRLLM]
    results = runner.run_all(backends)
    for r in results:
        _print_result(r)


@app.command("import-gpu")
def cmd_import_gpu(
    json_path: Optional[Path] = typer.Option(None, "--json-path", help="Path to Colab JSON."),  # noqa: UP007
) -> None:
    """Import a GPU run result, or record a placeholder N/A result if none provided."""
    settings = get_settings()
    runner = BenchmarkRunner(settings)
    result = runner.import_gpu_result(json_path)
    _print_result(result)


@app.command("report")
def cmd_report() -> None:
    """Print a comparison table from all saved results."""
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    df = summarize(results_dir)
    if df.empty:
        typer.echo("No results found.  Run `airllm-bench run --backend <name>` first.")
        raise typer.Exit(1)
    cols = [
        "backend", "status", "load_time_s", "generate_time_s",
        "total_runtime_s", "tokens_per_s", "peak_process_rss_mb",
    ]
    existing = [c for c in cols if c in df.columns]
    typer.echo(df[existing].to_string(index=False))

    table_path = write_comparison_table(df, results_dir)
    typer.echo(f"\n{render_comparison_markdown(df)}")
    typer.echo(f"\nTable saved to: {table_path}")

    assets_dir = Path(settings.assets_dir)
    paths = plot_comparison(df, assets_dir)
    if paths:
        names = ", ".join(p.name for p in paths)
        typer.echo(f"Charts saved to: {assets_dir} ({names})")


@app.command("host-spec")
def cmd_host_spec() -> None:
    """Print and save the current host hardware specification."""
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    path = write_host_spec(results_dir)
    spec = capture_host_spec()
    typer.echo(json.dumps(spec, indent=2))
    typer.echo(f"\nSaved to {path}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _print_result(result) -> None:  # noqa: ANN001
    typer.echo(
        f"\n{'─' * 50}\n"
        f"  Backend : {result.backend}\n"
        f"  Status  : {result.status}\n"
        f"  Model   : {result.model_id}\n"
        f"  Load    : {result.load_time_s:.2f}s\n" if result.load_time_s else ""
        f"  Generate: {result.generate_time_s:.2f}s\n" if result.generate_time_s else ""
        f"  Tok/s   : {result.tokens_per_s:.2f}\n" if result.tokens_per_s else ""
        f"  Peak RSS: {result.peak_process_rss_mb:.1f} MB\n" if result.peak_process_rss_mb else ""
        f"  Reason  : {result.failure_reason}\n" if result.failure_reason else ""
        f"  Preview : {result.output_preview!r}\n" if result.output_preview else ""
        f"{'─' * 50}\n"
    )


if __name__ == "__main__":
    app()
