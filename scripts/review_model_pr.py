from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pyhazards.model_catalog import (
    MODEL_PR_MARKER,
    MODEL_REVIEW_MARKER,
    NON_CATALOG_MODELS,
    builder_contract_issues,
    card_by_registry_name,
    load_model_cards,
    model_catalog_alignment_issues,
    run_smoke_test,
    source_contract_issues,
    touched_card_names,
)


REQUIRED_PR_SECTIONS = [
    "Model Summary",
    "Hazard Scenario",
    "Registry Name",
    "Paper / Source",
    "Smoke Test",
    "Parity Notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review a model contribution PR using the PyHazards model catalog."
    )
    parser.add_argument("--base-sha", help="Base commit SHA for git diff.")
    parser.add_argument("--event", help="Path to the GitHub event JSON payload.")
    parser.add_argument("--report-json", required=True, help="Output path for JSON report.")
    parser.add_argument("--report-md", required=True, help="Output path for Markdown report.")
    return parser.parse_args()


def git_changed_files(base_sha: Optional[str]) -> List[str]:
    revision = "{base}...HEAD".format(base=base_sha) if base_sha else "HEAD~1...HEAD"
    result = subprocess.run(
        ["git", "diff", "--name-only", revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_event(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_sections(body: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    if not body:
        return sections

    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def is_model_pr(changed_files: Sequence[str], body: str) -> bool:
    if any(path.startswith("pyhazards/model_cards/") for path in changed_files):
        return True
    return "- [x] Model contribution" in body


def markdown_report(
    *,
    status: str,
    summary: str,
    changed_files: Sequence[str],
    touched_models: Sequence[str],
    blockers: Sequence[str],
    warnings: Sequence[str],
) -> str:
    lines: List[str] = [
        MODEL_REVIEW_MARKER,
        "",
        "## PyHazards Model PR Review",
        "",
        "Status: **{status}**".format(status=status.upper()),
        "",
        summary,
        "",
    ]

    if touched_models:
        lines.extend(
            [
                "Touched models:",
                "",
            ]
        )
        for name in touched_models:
            lines.append("- ``{name}``".format(name=name))
        lines.append("")

    if blockers:
        lines.extend(["Blocking issues:", ""])
        for issue in blockers:
            lines.append("- {issue}".format(issue=issue))
        lines.append("")

    if warnings:
        lines.extend(["Warnings:", ""])
        for issue in warnings:
            lines.append("- {issue}".format(issue=issue))
        lines.append("")

    if changed_files:
        lines.extend(["Changed files reviewed:", ""])
        for path in changed_files:
            lines.append("- ``{path}``".format(path=path))
        lines.append("")

    lines.extend(
        [
            "Automation notes:",
            "",
            "- Passing model PRs are merged automatically by the PR bot workflow.",
            "- The public model tables and module pages are generated from ``pyhazards/model_cards/*.yaml``.",
            "- Use ``python scripts/smoke_test_models.py --models {models}`` locally before pushing.".format(
                models=" ".join(touched_models) if touched_models else "<model_name>"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report_json: Path, report_md: Path, payload: Dict[str, object], markdown: str) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(markdown, encoding="utf-8")


def main() -> int:
    args = parse_args()
    event = load_event(args.event)
    pull_request = event.get("pull_request", {}) if isinstance(event, dict) else {}
    body = ""
    draft = False
    if isinstance(pull_request, dict):
        body = str(pull_request.get("body") or "")
        draft = bool(pull_request.get("draft"))

    changed_files = git_changed_files(args.base_sha)
    model_pr = is_model_pr(changed_files, body)

    if not model_pr:
        payload = {
            "status": "skip",
            "is_model_pr": False,
            "summary": "No catalog-backed model contribution detected in this PR.",
            "models": [],
            "blockers": [],
            "warnings": [],
        }
        markdown = markdown_report(
            status="skip",
            summary=payload["summary"],
            changed_files=changed_files,
            touched_models=[],
            blockers=[],
            warnings=[],
        )
        write_reports(Path(args.report_json), Path(args.report_md), payload, markdown)
        return 0

    blockers: List[str] = []
    warnings: List[str] = []

    if MODEL_PR_MARKER not in body:
        warnings.append(
            "The PR template marker is missing. Use `.github/PULL_REQUEST_TEMPLATE.md` so the bot can parse the model metadata reliably."
        )

    sections = extract_sections(body)
    for section in REQUIRED_PR_SECTIONS:
        if not sections.get(section):
            blockers.append(
                "Fill in the PR template section `## {section}` with project-specific details.".format(
                    section=section
                )
            )

    try:
        cards = load_model_cards()
    except Exception as exc:  # pragma: no cover - exercised via CLI path
        blockers.append("Unable to load model cards: {error}".format(error=exc))
        cards = []

    touched_names = touched_card_names(cards, changed_files) if cards else []
    mapping = card_by_registry_name(cards) if cards else {}

    if not touched_names:
        blockers.append(
            "Model PRs must add or update at least one YAML card under `pyhazards/model_cards/`."
        )

    if sections.get("Registry Name"):
        registry_text = sections["Registry Name"]
        for name in touched_names:
            if name not in registry_text:
                blockers.append(
                    "The `Registry Name` section should mention `{name}` so the described model matches the implementation under review.".format(
                        name=name
                    )
                )

    if cards:
        blockers.extend(model_catalog_alignment_issues(cards))
        for name in touched_names:
            if name not in mapping:
                blockers.append(
                    "No valid model card was loaded for `{name}`. Check the YAML filename and schema.".format(
                        name=name
                    )
                )
                continue

            card = mapping[name]
            if card.model_name in NON_CATALOG_MODELS:
                continue

            blockers.extend(builder_contract_issues(card))
            warnings.extend(source_contract_issues(card))

            try:
                smoke = run_smoke_test(card)
            except Exception as exc:  # pragma: no cover - exercised via CLI path
                blockers.append(
                    "Smoke test failed for `{name}`: {error}".format(name=name, error=exc)
                )
            else:
                if not smoke["ok"]:
                    blockers.append(
                        "Smoke test shape mismatch for `{name}`: expected {expected}, got {actual}.".format(
                            name=name,
                            expected=smoke["expected_shape"],
                            actual=smoke["actual_shape"],
                        )
                    )

            hazard_section = sections.get("Hazard Scenario", "")
            if card.hazard not in hazard_section:
                warnings.append(
                    "The `Hazard Scenario` section should mention `{hazard}` so the generated model tables land in the intended section.".format(
                        hazard=card.hazard
                    )
                )

    if draft:
        warnings.append("This PR is still marked as draft, so the bot will not merge it yet.")

    status = "block" if blockers else "pass"
    summary = (
        "The PR satisfies the current PyHazards model contract and the synthetic smoke test."
        if status == "pass"
        else "The PR is missing one or more blocking model-contract requirements."
    )

    payload = {
        "status": status,
        "is_model_pr": True,
        "summary": summary,
        "models": touched_names,
        "blockers": blockers,
        "warnings": warnings,
        "draft": draft,
    }
    markdown = markdown_report(
        status=status,
        summary=summary,
        changed_files=changed_files,
        touched_models=touched_names,
        blockers=blockers,
        warnings=warnings,
    )
    write_reports(Path(args.report_json), Path(args.report_md), payload, markdown)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
