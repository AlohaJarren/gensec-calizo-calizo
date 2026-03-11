# HW9 - Promptfoo Vulnerable Target and Red-Team Setup

This homework builds a small local LLM application and then tests it with Promptfoo.

The goal is simple. I want to show a clean example of how an LLM application can become vulnerable when user-controlled text is placed in a high-trust instruction context. Then I want to show how Promptfoo can find that weakness in a structured way.

## Commit 1 - vulnerable local target

This commit adds a small Flask application.

The app exposes one local endpoint at `/chat`. It accepts a prompt and returns a model response.

The vulnerability is intentional. The application places attacker-controlled text inside the model's system instruction block instead of treating it like normal user input. That design choice creates a clear prompt-injection target for later testing.

### Setup

cd hw9
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py

---

# Commit 2 - Promptfoo config and runner

Commit 2 - Promptfoo config and runner

This commit adds the Promptfoo configuration and the runner script.

The configuration points Promptfoo to the local Flask target. The scan stays narrow on purpose. I want it to be easy to explain, easy to reproduce, and strong enough to show the core weakness clearly.

This red-team setup focuses on prompt extraction, system prompt override, and a custom policy check. The policy is straightforward: the model must not reveal internal notes, hidden prompts, or internal tokens.

## What changed

- `promptfooconfig.yaml` connects Promptfoo to the local /chat endpoint
- `run_promptfoo.sh` runs the red-team workflow and report
- the scan is scoped to a small set of checks that match the weakness in Commit 1

## Why these checks

The app from Commit 1 is specifically vulnerable to direct prompt injection and prompt leakage, so this config focuses on:

- `prompt-extraction`
- `system-prompt-override`
- a custom `policy` that says the app must not reveal internal notes or tokens

## Prerequisites

- The Flask app from Commit 1 should be running locally.
- Node.js 20 or newer should be installed.
- `ANTHROPIC_API_KEY` is exported or present in your shell environment

## Run

cd hw9
./run_promptfoo.sh both

## Open the report only

cd hw9
./run_promptfoo.sh report

---

# Commit 3 - local smoke test and evidence logging

This commit adds a short local workflow for quick validation.

I do not want to wait for a full Promptfoo scan every time I make a small change. Because of that, this commit adds one small probe script and one runner script.

demo_run.sh starts the Flask app, waits for the local endpoint to respond, runs one prompt-injection probe, saves the model output, and then shuts the app down.

demo_probe.py sends the probe request and writes the response to artifacts/manual_probe.json.

This gives me a fast baseline before the final red-team run.

## Run the smoke test

cd hw9
source .venv/bin/activate
./demo_run.sh

## Saved files

- artifacts/manual_probe.json
- artifacts/app.log

---

# Commit 4 - pinned Promptfoo version and final demo runner

This commit makes the final workflow more stable and easier to demonstrate.

I pinned Promptfoo in package.json so the project has a known local version. Then I updated run_promptfoo.sh so it prefers the local install when it exists.

I also added final_redteam.sh. This script starts the Flask app, waits for the endpoint to come up, runs the Promptfoo red-team scan, saves the logs, and can open the report at the end.

## One-time Node setup

cd hw9
npm install

## Final scan only

cd hw9
source .venv/bin/activate
./final_redteam.sh run

## Final scan and report

cd hw9
source .venv/bin/activate
./final_redteam.sh both

## Saved files

- artifacts/app_final.log
- artifacts/promptfoo_redteam.log

# Project Direction (Small Project)

First, I create a vulnerable LLM application. Then, I use Promptfoo to test it in a repeatable way. That gives me a focused demo, a clean commit history, and a project that reflects what we have been learning in CS 575 about LLM security, evaluation, and exploitation workflows.
