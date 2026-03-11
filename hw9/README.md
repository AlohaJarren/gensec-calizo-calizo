# HW9 Commit 1 - vulnerable Promptfoo target

This commit adds a small local Flask app that Promptfoo will attack in later commits.

## Why it is vulnerable

The app incorrectly pastes attacker-controlled text into the model's system instruction block instead of treating it as normal user input.

## Setup

```bash
cd hw9
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```
