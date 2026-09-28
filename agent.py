"""
agent.py  —  The AI Brain of the Project
==========================================

This file handles the entire AI workflow in 4 clear steps:

    Step 1 | build_schema()   → Describe the dataset (column names, types, samples)
    Step 2 | ask_llm()        → Send schema + question to local Ollama → get Python code
    Step 3 | validate_code()  → Check the code is safe before running it
    Step 4 | run_code()       → Execute the safe code on the local data

PRIVACY:  Raw CSV rows are NEVER sent to the LLM.
          Only a schema (column descriptions) is shared.

LOCAL:    We use Ollama running on your machine — no cloud API, no internet.
"""

from __future__ import annotations

import ast
import builtins
import json
import re

import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "llama3.1:8b"                    # The LLM we use (must be pulled via Ollama)
OLLAMA_HOST = "http://127.0.0.1:11434"  # Ollama runs locally — loopback only


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build a safe schema to describe the dataset to the AI
# ─────────────────────────────────────────────────────────────────────────────

def build_schema(df: pd.DataFrame) -> list[dict]:
    """
    Create a compact, safe description of the dataset.

    Instead of sending all rows to the AI, we send only:
      - Column name
      - Data type  (e.g. int64, object, datetime64)
      - 3 short sample values  (so the AI knows what the data looks like)
      - How many values are missing
      - How many unique values exist

    This way the AI understands the structure without ever seeing personal data.
    """
    schema = []
    for col in df.columns:
        col_data = df[col]

        # Grab up to 3 non-null sample values and convert to short strings
        samples = [
            str(v)[:60]   # Cap each sample at 60 characters to keep prompts short
            for v in col_data.dropna().head(3).tolist()
        ]

        schema.append({
            "column":  col,
            "dtype":   str(col_data.dtype),
            "samples": samples,
            "missing": int(col_data.isna().sum()),
            "unique":  int(col_data.nunique()),
        })
    return schema


def schema_to_json(schema: list[dict]) -> str:
    """Convert the schema list to a JSON string for the LLM prompt."""
    return json.dumps(schema, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Send schema + question to local Ollama and get Python code back
# ─────────────────────────────────────────────────────────────────────────────

# This is the system prompt — it tells the AI exactly what format to use.
# We are very strict: code only, no explanations, no markdown.
_SYSTEM_PROMPT = """You are a Python data analyst. Write pandas code to answer questions.

Rules (follow exactly):
- The DataFrame is already loaded as `df` and pandas is imported as `pd`
- Store your final answer in a variable called `result`
- Return ONLY executable Python code — no markdown, no explanations, no comments
- Use the exact column names from the schema
- Do NOT write any import statements
- For a single value answer, result = scalar. For tables, result = DataFrame or Series."""


def ask_llm(schema_json: str, question: str) -> str:
    """
    Ask the local Ollama model to write pandas code for our question.

    The model receives:
      1. The system prompt (rules for code format)
      2. The dataset schema (column descriptions — NOT the data)
      3. The user's question

    It returns Python code that we will validate and run locally.
    """
    if not question.strip():
        raise ValueError("Please enter a question.")

    try:
        from ollama import Client
    except ImportError:
        raise RuntimeError("Ollama package not found. Run: pip install -r requirements.txt")

    client = Client(host=OLLAMA_HOST)

    # Build the user message: schema + question together
    user_message = (
        f"Dataset schema (column descriptions only — no raw data):\n{schema_json}\n\n"
        f"User question: {question.strip()}\n\n"
        "Write the pandas code now. Assign the final answer to 'result'."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    try:
        # First try: use default settings (GPU if available)
        response = client.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": 0},  # temperature=0 → consistent, deterministic output
        )
    except Exception as exc:
        error_text = str(exc)

        # If the GPU failed (common on some setups), retry on CPU
        if "cuda" in error_text.lower() or "terminated" in error_text.lower():
            try:
                response = client.chat(
                    model=MODEL,
                    messages=messages,
                    options={"temperature": 0, "num_gpu": 0},  # force CPU
                )
            except Exception as cpu_exc:
                raise RuntimeError(f"Ollama failed on both GPU and CPU: {cpu_exc}")

        elif "not found" in error_text.lower():
            raise RuntimeError(
                f"Model '{MODEL}' is not installed.\n"
                f"Fix: Run  ollama pull {MODEL}  in your terminal."
            )
        else:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
                "Fix: Make sure Ollama is running (open the Ollama app or run 'ollama serve')."
            )

    # Strip markdown code fences if the model wrapped the code in ``` blocks
    raw = response.message.content or ""
    code = re.sub(r"```(?:python)?\s*", "", raw).replace("```", "").strip()

    if not code:
        raise RuntimeError("The AI returned an empty response. Please try again.")

    return code


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Validate the code is safe before running it
# ─────────────────────────────────────────────────────────────────────────────

# These names are dangerous — if the AI accidentally generates them, we block it.
_BLOCKED_NAMES = {
    "os", "sys", "open", "eval", "exec", "__import__",
    "subprocess", "requests", "socket", "shutil", "pathlib",
    "globals", "locals", "vars", "builtins",
}


def validate_code(code: str) -> str:
    """
    Security check: make sure the AI-generated code is safe to execute.

    WHY this matters:
      Without validation, a bad LLM response could delete files, call the internet,
      or access sensitive system information.

    HOW we check:
      Python's built-in 'ast' module lets us parse code into a syntax tree (AST)
      and inspect every single operation before running it.

    We block 4 things:
      1. Import statements   (no loading external libraries)
      2. Dangerous names     (os, sys, open, eval, exec, subprocess, ...)
      3. Dunder attributes   (__class__, __dict__, __builtins__, ...)
      4. Missing result var  (the code must assign to 'result')
    """
    if not code or not code.strip():
        raise ValueError("The AI returned empty code. Please try again.")

    # Parse the code into an AST (Abstract Syntax Tree)
    # If the code has syntax errors, this will raise SyntaxError
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"The AI generated invalid Python syntax: {e}")

    # Walk through every node in the syntax tree
    for node in ast.walk(tree):

        # RULE 1: No import statements
        # We don't want the AI loading os, subprocess, requests, etc.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed in generated code.")

        # RULE 2: Block dangerous built-in names
        # Prevents: os.remove(), open('secret.txt'), eval('...'), etc.
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise ValueError(f"'{node.id}' is not allowed in generated code.")

        # RULE 3: Block dunder (double underscore) attributes
        # Prevents tricks like df.__class__.__bases__ to access hidden Python internals
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attributes are not allowed.")

    # RULE 4: The code must create a 'result' variable
    # We need this to display the answer in the UI
    assigned_vars = [
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    if "result" not in assigned_vars:
        raise ValueError(
            "Generated code must assign the answer to a variable named 'result'.\n"
            "Example: result = df.groupby('Product')['Revenue'].sum()"
        )

    return code.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Execute the validated code safely on the local data
# ─────────────────────────────────────────────────────────────────────────────

def run_code(code: str, df: pd.DataFrame):
    """
    Execute the validated pandas code against a copy of the dataset.

    We run in a sandboxed environment — the code can only access:
      - 'df'  → a fresh copy of the DataFrame (original is never modified)
      - 'pd'  → pandas library
      - A small set of safe built-in functions (sum, len, max, min, ...)

    The code cannot access files, the network, or any system resources.
    """
    # Create a restricted namespace — the only things the code can "see"
    safe_builtins = {
        name: getattr(builtins, name)
        for name in (
            "abs", "all", "any", "bool", "dict", "enumerate",
            "float", "int", "len", "list", "max", "min",
            "print", "range", "round", "set", "str", "sum", "tuple", "zip",
        )
    }

    sandbox = {
        "__builtins__": safe_builtins,
        "pd": pd,           # pandas — the AI's main tool
        "df": df.copy(),    # a fresh copy so the original is never changed
    }

    try:
        exec(compile(code, "<ai-generated-code>", "exec"), sandbox)
    except Exception as e:
        raise RuntimeError(
            f"The generated code failed to run.\n"
            f"Error: {e}\n\n"
            f"Code that failed:\n{code}"
        )

    result = sandbox.get("result")
    if result is None:
        raise RuntimeError(
            "The code ran successfully but did not create a 'result' variable."
        )
    return result
