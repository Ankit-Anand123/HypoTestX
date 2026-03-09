"""
HypoTestX Command-Line Interface
=================================

Usage
-----
hypotestx analyze --help
hypotestx analyze data.csv "Do males earn more than females?"
hypotestx analyze data.csv "Is age correlated with salary?" --backend gemini --api-key AIza...
hypotestx analyze data.csv "Compare regions" --backend ollama --model phi4 --verbose
hypotestx version
hypotestx backends
"""
from __future__ import annotations

import argparse
import sys
import os


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_csv(path: str):
    """Load a CSV file into a dict-of-lists (no pandas required)."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        print(f"[error] '{path}' is empty or has no data rows.", file=sys.stderr)
        sys.exit(1)
    # Build dict-of-lists; auto-cast numeric columns
    cols: dict = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(v)
    return cols


def _try_pandas(path: str):
    """Try to load with pandas for richer dtype inference; fall back to CSV."""
    try:
        import pandas as pd  # type: ignore
        return pd.read_csv(path)
    except ImportError:
        return _load_csv(path)


# --------------------------------------------------------------------------- #
# Sub-commands                                                                 #
# --------------------------------------------------------------------------- #

def cmd_analyze(args: argparse.Namespace) -> None:
    from hypotestx.core.engine import analyze

    if not os.path.isfile(args.file):
        print(f"[error] File not found: '{args.file}'", file=sys.stderr)
        sys.exit(1)

    df = _try_pandas(args.file)

    backend_kwargs: dict = {}
    if args.api_key:
        backend_kwargs["api_key"] = args.api_key
    if args.model:
        backend_kwargs["model"] = args.model
    if args.host:
        backend_kwargs["host"] = args.host

    try:
        result = analyze(
            df,
            args.question,
            backend=args.backend,
            alpha=args.alpha,
            verbose=args.verbose,
            **backend_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "summary":
        print(result.summary())
    elif args.format == "json":
        import json
        d = result.to_dict() if hasattr(result, "to_dict") else vars(result)
        print(json.dumps(d, indent=2, default=str))
    elif args.format == "apa":
        try:
            from hypotestx.reporting.generator import apa_report
            print(apa_report(result))
        except Exception:
            print(result.summary())
    else:
        print(result.summary())


def cmd_version(_args) -> None:
    import hypotestx
    print(f"HypoTestX {hypotestx.__version__}")


def cmd_backends(_args) -> None:
    print("Available LLM backends:")
    print()
    print("  none / fallback   Built-in regex router (default, zero deps, offline)")
    print("  ollama            Local Ollama server  (ollama.com, free, offline)")
    print("  gemini            Google Gemini API    (free tier: 1500 req/day)")
    print("  groq              Groq Cloud API       (free tier, OpenAI-compatible)")
    print("  openai            OpenAI API           (paid)")
    print("  together          Together AI          (free tier)")
    print("  mistral           Mistral AI           (free tier)")
    print("  perplexity        Perplexity AI        (paid)")
    print("  huggingface / hf  HuggingFace Inference API or local transformers")
    print()
    print("Pass --api-key / --model / --host to supply backend credentials.")


# --------------------------------------------------------------------------- #
# Argument parser                                                              #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypotestx",
        description="HypoTestX — Natural Language Hypothesis Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hypotestx analyze data.csv "Do males earn more than females?"
  hypotestx analyze data.csv "Is age correlated with salary?" --verbose
  hypotestx analyze data.csv "Compare regions" --backend gemini --api-key AIza...
  hypotestx analyze data.csv "Is there an association?" --format apa
  hypotestx backends
  hypotestx version
""",
    )

    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ── analyze ──────────────────────────────────────────────────────────────
    an = sub.add_parser(
        "analyze",
        help="Run a hypothesis test from a plain-English question and a CSV file.",
    )
    an.add_argument("file", help="Path to CSV data file")
    an.add_argument("question", help="Plain-English hypothesis question")
    an.add_argument(
        "--backend", "-b",
        default=None,
        metavar="BACKEND",
        help="LLM backend to use: none/fallback, ollama, gemini, groq, openai, "
             "together, mistral, huggingface (default: fallback)",
    )
    an.add_argument(
        "--api-key", "-k",
        default=None,
        dest="api_key",
        metavar="KEY",
        help="API key for cloud LLM backends",
    )
    an.add_argument(
        "--model", "-m",
        default=None,
        metavar="MODEL",
        help="Model name override (e.g. phi4, gemini-1.5-pro, gpt-4o-mini)",
    )
    an.add_argument(
        "--host",
        default=None,
        metavar="URL",
        help="Custom base URL for Ollama or OpenAI-compatible servers",
    )
    an.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Significance level (default: 0.05)",
    )
    an.add_argument(
        "--format", "-f",
        choices=["summary", "json", "apa"],
        default="summary",
        help="Output format: summary (default), json, or apa",
    )
    an.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show routing info (chosen test, LLM reasoning)",
    )
    an.set_defaults(func=cmd_analyze)

    # ── version ───────────────────────────────────────────────────────────────
    ver = sub.add_parser("version", help="Show HypoTestX version.")
    ver.set_defaults(func=cmd_version)

    # ── backends ──────────────────────────────────────────────────────────────
    bk = sub.add_parser("backends", help="List available LLM backends.")
    bk.set_defaults(func=cmd_backends)

    return parser


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
