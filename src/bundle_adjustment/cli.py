"""Command-line interface for supported bundle-adjustment workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .io import load_colmap_text, read_result_metadata, save_result
from .models import OptimizationOptions
from .optimizer import optimize
from .validation import validate_problem


def _options_from_arguments(arguments: argparse.Namespace) -> OptimizationOptions:
    return OptimizationOptions(
        max_iterations=arguments.max_iterations,
        initial_damping=arguments.initial_damping,
        loss=arguments.loss,
        huber_delta=arguments.huber_delta,
        fixed_camera_indices=tuple(arguments.fix_camera),
        fixed_point_indices=tuple(arguments.fix_point),
        auto_gauge_fix=not arguments.no_auto_gauge_fix,
    )


def _problem_summary(
    problem_name: str, cameras: int, points: int, observations: int
) -> dict[str, int | str]:
    return {
        "problem_name": problem_name,
        "cameras": cameras,
        "points": points,
        "observations": observations,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the parser used by the ``bundle-adjust`` console command."""
    parser = argparse.ArgumentParser(
        prog="bundle-adjust",
        description="Validate and refine initialized COLMAP text reconstructions.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate_parser = subcommands.add_parser("validate", help="validate a COLMAP text model")
    validate_parser.add_argument("model_directory", type=Path)

    optimize_parser = subcommands.add_parser("optimize", help="optimize a COLMAP text model")
    optimize_parser.add_argument("model_directory", type=Path)
    optimize_parser.add_argument("output_directory", type=Path)
    optimize_parser.add_argument("--max-iterations", type=int, default=50)
    optimize_parser.add_argument("--initial-damping", type=float, default=1e-3)
    optimize_parser.add_argument("--loss", choices=("squared", "huber"), default="huber")
    optimize_parser.add_argument("--huber-delta", type=float, default=2.0)
    optimize_parser.add_argument("--fix-camera", type=int, action="append", default=[])
    optimize_parser.add_argument("--fix-point", type=int, action="append", default=[])
    optimize_parser.add_argument("--no-auto-gauge-fix", action="store_true")
    optimize_parser.add_argument("--overwrite", action="store_true")

    report_parser = subcommands.add_parser("report", help="print an optimization result summary")
    report_parser.add_argument("output_directory", type=Path)
    report_parser.add_argument("--json", action="store_true", help="emit full JSON metadata")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a conventional process exit status."""
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "validate":
            problem = load_colmap_text(parsed.model_directory)
            validate_problem(problem, OptimizationOptions())
            print(
                json.dumps(
                    _problem_summary(
                        problem.name,
                        problem.num_cameras,
                        problem.num_points,
                        len(problem.observations),
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if parsed.command == "optimize":
            problem = load_colmap_text(parsed.model_directory)
            result = optimize(problem, _options_from_arguments(parsed))
            save_result(result, parsed.output_directory, overwrite=parsed.overwrite)
            print(
                json.dumps(
                    {
                        "converged": result.converged,
                        "final_cost": result.diagnostics.final_cost,
                        "final_rmse": result.diagnostics.final_rmse,
                        "output_directory": str(parsed.output_directory),
                        "termination_reason": result.termination_reason,
                    },
                    sort_keys=True,
                )
            )
            return 0 if result.converged else 1
        if parsed.command == "report":
            metadata = read_result_metadata(parsed.output_directory)
            if parsed.json:
                print(json.dumps(metadata, indent=2, sort_keys=True))
            else:
                diagnostics = metadata["diagnostics"]
                cost_summary = (
                    f"Cost: {diagnostics['initial_cost']:.6g} -> {diagnostics['final_cost']:.6g}"
                )
                rmse_summary = (
                    f"RMSE: {diagnostics['initial_rmse']:.6g} -> {diagnostics['final_rmse']:.6g}"
                )
                print(
                    "\n".join(
                        (
                            f"Problem: {metadata['problem_name']}",
                            f"Converged: {metadata['converged']}",
                            f"Termination: {metadata['termination_reason']}",
                            cost_summary,
                            rmse_summary,
                        )
                    )
                )
            return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    raise AssertionError(f"Unexpected command: {parsed.command}")
