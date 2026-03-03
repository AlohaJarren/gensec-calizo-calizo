#!/usr/bin/env python3
"""Statically triage Python code and optionally add a Gemini verdict."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

from google import genai

URL_RE = re.compile(r"https?://[^\s'\"<>]+")
HEX_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){3,}")
B64_RE = re.compile(r"\b[A-Za-z0-9+/]{16,}={0,2}\b")
CHR_RE = re.compile(r"(?:chr\(\d+\)\s*\+\s*){2,}chr\(\d+\)")

WEIGHTS = {"low": 1, "medium": 2, "high": 3}
RULES = {
    "eval": ("dynamic_execution", "high", "Uses eval."),
    "exec": ("dynamic_execution", "high", "Uses exec."),
    "compile": ("dynamic_execution", "medium", "Builds code objects dynamically."),
    "__import__": ("dynamic_import", "medium", "Imports modules dynamically."),
    "requests.get": ("network_access", "medium", "Makes an outbound HTTP GET request."),
    "requests.post": ("network_access", "medium", "Makes an outbound HTTP POST request."),
    "requests.put": ("network_access", "medium", "Makes an outbound HTTP PUT request."),
    "requests.delete": ("network_access", "medium", "Makes an outbound HTTP DELETE request."),
    "urllib.request.urlopen": ("network_access", "medium", "Opens a remote URL."),
    "subprocess.run": ("process_launch", "high", "Launches a child process."),
    "subprocess.Popen": ("process_launch", "high", "Launches a child process."),
    "subprocess.call": ("process_launch", "high", "Launches a child process."),
    "subprocess.check_call": ("process_launch", "high", "Launches a child process."),
    "subprocess.check_output": ("process_launch", "high", "Launches a child process."),
    "os.system": ("process_launch", "high", "Executes a shell command."),
    "os.popen": ("process_launch", "high", "Executes a shell command."),
    "os.getenv": ("environment_access", "medium", "Reads an environment variable."),
    "base64.b64decode": ("obfuscation_or_decode", "medium", "Decodes Base64 content."),
    "binascii.unhexlify": ("obfuscation_or_decode", "medium", "Decodes hex content."),
    "codecs.decode": ("obfuscation_or_decode", "medium", "Decodes encoded content."),
}


def dotted_name(node: ast.AST) -> str | None:
    """Return a dotted name for a Name/Attribute AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def open_mode(node: ast.Call) -> str | None:
    """Extract the string mode from an open-like call if present."""
    values = []
    if len(node.args) >= 2:
        values.append(node.args[1])
    values.extend(k.value for k in node.keywords if k.arg == "mode")
    for value in values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


class Analyzer(ast.NodeVisitor):
    """Collect suspicious signals from Python source."""

    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()
        self.imports: set[str] = set()
        self.urls = sorted(set(URL_RE.findall(text)))
        self.signals: list[tuple[str, str, str, str, int, str]] = []

    def add(self, category: str, severity: str, rule: str, reason: str, line: int | None) -> None:
        """Add one signal if it is not already present."""
        line_no = line or 0
        evidence = self.lines[line_no - 1].strip() if 1 <= line_no <= len(self.lines) else ""
        item = (category, severity, rule, reason, line_no, evidence)
        if item not in self.signals:
            self.signals.append(item)

    def visit_Import(self, node: ast.Import) -> None:
        """Collect imported modules."""
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Collect imported modules from from-import statements."""
        if node.module:
            self.imports.add(node.module.split(".")[0])
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Detect os.environ[...] access."""
        if dotted_name(node.value) == "os.environ":
            self.add(
                "environment_access",
                "medium",
                "os.environ[]",
                "Reads an environment variable via os.environ[...].",
                getattr(node, "lineno", None),
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect risky calls and write-capable file access."""
        name = dotted_name(node.func)

        if name in RULES:
            category, severity, reason = RULES[name]
            self.add(category, severity, name, reason, getattr(node, "lineno", None))

        if name == "open":
            mode = open_mode(node)
            if mode and any(flag in mode for flag in "wax+"):
                self.add(
                    "file_write",
                    "medium",
                    "open-write",
                    f"Opens a file in write-capable mode ({mode}).",
                    getattr(node, "lineno", None),
                )

        if name in {"Path.write_text", "Path.write_bytes"}:
            self.add(
                "file_write",
                "medium",
                name,
                "Writes content to the filesystem.",
                getattr(node, "lineno", None),
            )

        if name and name.endswith(".open"):
            mode = open_mode(node)
            if mode and any(flag in mode for flag in "wax+"):
                self.add(
                    "file_write",
                    "medium",
                    f"{name}-write",
                    f"Opens a path object in write-capable mode ({mode}).",
                    getattr(node, "lineno", None),
                )

        self.generic_visit(node)


def discover(target: Path) -> list[Path]:
    """Return Python files for a file or directory target."""
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    if target.is_dir():
        return sorted(p for p in target.rglob("*.py") if p.is_file())
    return []


def label(score: int) -> str:
    """Map a numeric score to a risk label."""
    if score == 0:
        return "clean"
    if score <= 3:
        return "low"
    if score <= 7:
        return "medium"
    return "high"


def add_regex_signals(text: str, report: dict) -> None:
    """Add simple regex-based obfuscation hints."""
    lines = text.splitlines()

    def add(pattern: re.Pattern[str], rule: str, severity: str, reason: str) -> None:
        match = pattern.search(text)
        if not match:
            return
        line_no = text[: match.start()].count("\n") + 1
        evidence = lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else ""
        item = ("obfuscation_or_decode", severity, rule, reason, line_no, evidence)
        if item not in report["signals"]:
            report["signals"].append(item)

    add(HEX_RE, "hex-escape-pattern", "medium", "Contains repeated hex escape sequences.")
    add(B64_RE, "base64-like-string", "low", "Contains a long Base64-like token.")
    add(CHR_RE, "chr-chain", "medium", "Builds a string from many chr(...) calls.")


def analyze(path: Path) -> dict:
    """Analyze one Python file and return a compact report dictionary."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

    report = {"path": path, "imports": [], "urls": [], "signals": [], "syntax_error": None}

    try:
        tree = ast.parse(text, filename=str(path))
        analyzer = Analyzer(text)
        analyzer.visit(tree)
        report["imports"] = sorted(analyzer.imports)
        report["urls"] = analyzer.urls
        report["signals"] = analyzer.signals
    except SyntaxError as exc:
        report["syntax_error"] = f"{exc.msg} (line {exc.lineno})"

    add_regex_signals(text, report)
    report["signals"].sort(key=lambda s: (s[4], s[0], s[2]))
    report["score"] = sum(WEIGHTS[s[1]] for s in report["signals"])
    return report


def print_report(target: Path, reports: list[dict], show_clean: bool) -> None:
    """Print a concise static triage summary."""
    total_score = sum(r["score"] for r in reports)
    total_signals = sum(len(r["signals"]) for r in reports)

    print(f"Target: {target}")
    print(f"Files scanned: {len(reports)}")
    print(f"Total signals: {total_signals}")
    print(f"Overall risk: {label(total_score).upper()} (score={total_score})")

    if total_score == 0:
        print("\nNo suspicious static signals found.")
        print("Limitations: static-only analysis; runtime behavior is not observed.")
        return

    print("\nPer-file findings:")
    for report in reports:
        if not report["signals"] and not show_clean:
            continue

        print(f"\n- {report['path']}")
        print(f"  Risk: {label(report['score']).upper()} (score={report['score']})")

        if report["syntax_error"]:
            print(f"  Syntax error: {report['syntax_error']}")

        if report["imports"]:
            print(f"  Imports: {', '.join(report['imports'])}")

        if report["urls"]:
            print("  URLs:")
            for url in report["urls"]:
                print(f"    - {url}")

        if not report["signals"]:
            print("  Signals: none")
            continue

        print("  Signals:")
        for category, severity, rule, reason, line_no, evidence in report["signals"]:
            where = f"line {line_no}" if line_no else "line unknown"
            print(f"    - [{severity.upper()}] {category} via {rule} ({where})")
            print(f"      Reason: {reason}")
            if evidence:
                print(f"      Evidence: {evidence}")

    print("\nLimitations: static-only analysis; this version does not execute or emulate code.")


def build_llm_input(reports: list[dict]) -> str:
    """Build a compact JSON summary for Gemini."""
    payload = {
        "files": [
            {
                "path": str(report["path"]),
                "risk": label(report["score"]),
                "urls": report["urls"],
                "signals": [
                    {
                        "category": category,
                        "severity": severity,
                        "rule": rule,
                        "reason": reason,
                        "line": line_no,
                        "evidence": evidence,
                    }
                    for category, severity, rule, reason, line_no, evidence in report["signals"]
                ],
            }
            for report in reports
        ]
    }
    return json.dumps(payload, indent=2)


def parse_json_text(text: str) -> dict:
    """Parse model output as JSON, tolerating fenced code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
    return json.loads(cleaned)


def gemini_verdict(reports: list[dict], model: str) -> dict:
    """Ask Gemini for a concise JSON verdict over the static findings."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    prompt = (
        "You are a security analyst.\n"
        "Given the static triage findings below, return JSON with exactly these keys:\n"
        "{\n"
        '  "summary": "one short paragraph",\n'
        '  "risk_level": "clean|low|medium|high",\n'
        '  "likely_behaviors": ["short item"],\n'
        '  "top_evidence": ["short item"],\n'
        '  "limitations": ["short item"]\n'
        "}\n\n"
        "Do not add markdown fences.\n\n"
        "Static findings:\n"
        f"{build_llm_input(reports)}"
    )

    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()

    try:
        return parse_json_text(text)
    except Exception:
        return {
            "summary": text,
            "risk_level": "unknown",
            "likely_behaviors": [],
            "top_evidence": [],
            "limitations": ["The model did not return valid JSON."],
        }


def print_llm_verdict(verdict: dict, model: str) -> None:
    """Print the Gemini verdict in a compact, readable format."""
    print("\nGemini verdict:")
    print(f"  Model: {model}")
    print(f"  Risk level: {verdict.get('risk_level', 'unknown')}")
    print(f"  Summary: {verdict.get('summary', '')}")

    if verdict.get("likely_behaviors"):
        print("  Likely behaviors:")
        for item in verdict["likely_behaviors"]:
            print(f"    - {item}")

    if verdict.get("top_evidence"):
        print("  Top evidence:")
        for item in verdict["top_evidence"]:
            print(f"    - {item}")

    if verdict.get("limitations"):
        print("  LLM limitations:")
        for item in verdict["limitations"]:
            print(f"    - {item}")


def main() -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description="Statically triage Python code for suspicious signals.")
    parser.add_argument("target", help="Path to a .py file or a directory.")
    parser.add_argument("--show-clean", action="store_true", help="Show files with no signals.")
    parser.add_argument("--use-llm", action="store_true", help="Add a Gemini verdict on top of static findings.")
    parser.add_argument(
        "--model",
        default=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use with --use-llm.",
    )
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    files = discover(target)
    if not files:
        print(f"No Python files found at: {target}", file=sys.stderr)
        return 1

    reports = [analyze(path) for path in files]
    print_report(target, reports, args.show_clean)

    if args.use_llm:
        try:
            verdict = gemini_verdict(reports, args.model)
            print_llm_verdict(verdict, args.model)
        except Exception as exc:
            print(f"\nGemini error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
