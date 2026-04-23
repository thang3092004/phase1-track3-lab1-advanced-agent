from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from rich.progress import Progress
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
    openai_model: str = "gpt-3.5-turbo",
) -> None:
    if mode not in {"mock", "openai"}:
        raise typer.BadParameter("mode must be either 'mock' or 'openai'")

    examples = load_dataset(dataset)
    react = ReActAgent(mode=mode, model_name=openai_model)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, mode=mode, model_name=openai_model)
    
    with Progress() as progress:
        react_task = progress.add_task("[cyan]Running ReAct Agent...", total=len(examples))
        react_records = []
        for example in examples:
            react_records.append(react.run(example))
            progress.update(react_task, advance=1)
        
        reflexion_task = progress.add_task("[green]Running Reflexion Agent...", total=len(examples))
        reflexion_records = []
        for example in examples:
            reflexion_records.append(reflexion.run(example))
            progress.update(reflexion_task, advance=1)
    
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
