"""Invokes a Bedrock Agent across two consecutive turns sharing the same sessionId.

Parent lab: bedrock-agents-201.
AWS Services: Amazon Bedrock Agent Runtime (invoke_agent + enableTrace).
Generated artifacts: agent/audit/invoke-audit.json (turns, completion, traceSteps).
Mode: live (si AGENT_ID/AGENT_ALIAS_ID sont fournis, sinon audit en mode "skipped").
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# Distinguishable failure modes when the call fails.
AWS_ERROR_HINTS = {
    "AccessDeniedException": "Access denied: either the IAM policy lacks bedrock:InvokeModel, or model access is not enabled (Bedrock console > Model access).",
    "ModelTimeoutException": "Model timed out: reduce prompt size or max tokens.",
    "ResourceNotFoundException": "Agent, alias, or Knowledge Base not found: check identifier and region.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: invalid model ID or model requires an inference profile./us./global.) plutôt que l'ID direct.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
AGENT_ID = os.environ.get("TRAINING_AGENT_ID", "REPLACE_WITH_AGENT_ID")
AGENT_ALIAS_ID = os.environ.get("TRAINING_AGENT_ALIAS_ID", "REPLACE_WITH_ALIAS_ID")
OUTPUT_PATH = Path("agent/audit/invoke-audit.json")

client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


def call_agent(session_id: str, question: str) -> dict:
    # enableTrace=True tracks reasoning steps for auditability.
    started = time.perf_counter()
    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=question,
        enableTrace=True,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    completion_parts: list[str] = []
    trace_steps: int = 0
    # Completion stream: accumulate text chunks and count reasoning traces.
    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        if chunk.get("bytes"):
            completion_parts.append(chunk["bytes"].decode("utf-8", errors="ignore"))
        if "trace" in event:
            trace_steps += 1
    return {
        "question": question,
        "completion": "".join(completion_parts).strip(),
        "traceSteps": trace_steps,
        "latencyMs": latency_ms,
    }


def main() -> None:
    # UUID sessionId to maintain conversational memory across turns.
    session_id = str(uuid.uuid4())
    skipped = AGENT_ID.startswith("REPLACE_") or AGENT_ALIAS_ID.startswith("REPLACE_")
    turns = [
        call_agent(session_id, "What is the capital of France?"),
        call_agent(session_id, "And what is the capital of Germany?"),
    ]
    audit = {
        "agentId": AGENT_ID,
        "agentAliasId": AGENT_ALIAS_ID,
        "sessionId": session_id,
        "region": AWS_REGION,
        "turns": turns,
        "skipped": skipped,
    }
    # Pedagogical note if the agent has not been provisioned.
    if skipped:
        audit["note"] = (
            "No AGENT_ID provided: setup required (create_agent + prepare_agent + create_agent_alias)."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ClientError as exc:
        sys.exit(explain_aws_error(exc))
    except NoCredentialsError:
        sys.exit("No AWS credentials found: run `aws configure` or export AWS_PROFILE.")
    except EndpointConnectionError:
        sys.exit("AWS endpoint unreachable: verify AWS_REGION and network access.")
    except FileNotFoundError as exc:
        sys.exit(f"Expected file not found: {exc.filename}. Run the previous lab step first.")
