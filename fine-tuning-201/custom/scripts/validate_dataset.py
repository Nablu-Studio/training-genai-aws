"""Validates a JSONL dataset for Amazon Bedrock fine-tuning (valid JSON + prompt/completion keys).

Parent lab: fine-tuning-201.
AWS Services: none (local execution).
Generated artifacts: custom/datasets/dataset-quality-report.json (issues, lineCount, passed).
Mode: offline processing, no live AWS calls.
"""
import json
import re
from pathlib import Path


DATASET_PATH = Path("custom/datasets/training-samples.jsonl")
REPORT_PATH = Path("custom/datasets/dataset-quality-report.json")

JSONL_PATTERN = re.compile(r"^\{.*\}$")


def validate_dataset() -> dict:
    # Read line-by-line: collect validation issues without failing early.
    lines = [line for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    issues: list[str] = []
    for index, line in enumerate(lines, start=1):
        # Fast check: valid JSON lines start with `{` and end with `}`.
        if not JSONL_PATTERN.match(line):
            issues.append(f"line {index} not a JSON object")
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"line {index} invalid JSON: {exc.msg}")
            continue
        # Bedrock fine-tuning requires `prompt` and `completion` keys.
        if "prompt" not in sample or "completion" not in sample:
            issues.append(f"line {index} missing prompt or completion")
    return {
        "lineCount": len(lines),
        "issueCount": len(issues),
        "issues": issues[:5],
        "passed": len(issues) == 0,
    }


def main() -> None:
    # Write validation report with pass/fail verdict.
    report = validate_dataset()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
