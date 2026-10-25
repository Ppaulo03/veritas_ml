"""High-level study orchestration pipeline."""

from __future__ import annotations

from typing import Any

from .core.config import RUN_DIR, STUDY_DIR, ensure_runtime_dirs
from .data import build_dataset_context
from .engine import build_experiment_plan, execute_experiments, plan_summary, run_sanity_checks
from .evaluation import (
    assemble_study_result,
    persist_study_result,
    render_study,
    run_diagnostics,
    write_manifest,
)
from .features import prepare_feature_context


def run_study(render: bool = True) -> dict[str, Any]:
    """Execute the complete media GNN study pipeline from dataset loading to report generation."""
    ensure_runtime_dirs()
    dataset = build_dataset_context()
    plan = build_experiment_plan(dataset)
    print(
        "Dataset:",
        len(dataset.frame),
        "development | audit reservado:",
        dataset.metadata["n_audit_reserved"],
    )
    print("Plano:", plan_summary(plan, dataset))

    print("1/5 — preparando representações")
    feature_context = prepare_feature_context(dataset)

    print("2/5 — sanity checks")
    assert run_sanity_checks(dataset, feature_context, plan)

    print("3/5 — executando matriz + experimentos focais")
    experiment_result = execute_experiments(dataset, feature_context, plan)

    print("4/5 — diagnósticos e StudyResult")
    diagnostics = run_diagnostics(dataset, feature_context, experiment_result, plan)
    study_result = assemble_study_result(dataset, experiment_result, diagnostics)

    report_dir = STUDY_DIR / "report"
    persist_study_result(study_result, report_dir)

    figures = []
    if render:
        print("5/5 — apresentação")
        figures = render_study(study_result, report_dir)

    write_manifest(dataset, feature_context, experiment_result, study_result, diagnostics, plan)
    return {
        "dataset": dataset,
        "plan": plan,
        "features": feature_context,
        "experiments": experiment_result,
        "diagnostics": diagnostics,
        "study": study_result,
        "figures": figures,
        "run_dir": RUN_DIR,
    }
