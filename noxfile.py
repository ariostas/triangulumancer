from __future__ import annotations

import re
from pathlib import Path

import nox


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the Python test suite.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def readme(session: nox.Session) -> None:
    """
    Run the README's examples as doctests against a fresh install.
    """
    session.install(".")
    # doctest would read each closing code fence as expected output.
    text = Path("README.md").read_text(encoding="utf-8")
    examples = Path(session.create_tmp()) / "README.txt"
    examples.write_text(re.sub(r"(?m)^```.*$", "", text), encoding="utf-8")
    session.run("python", "-m", "doctest", "-o", "ELLIPSIS", str(examples))


@nox.session(venv_backend="none")
def docs(session: nox.Session) -> None:
    """
    Build the Rust documentation as docs.rs does, with KaTeX.
    """
    session.run(
        "cargo",
        "doc",
        "--no-deps",
        *session.posargs,
        env={"RUSTDOCFLAGS": "--html-in-header docs-header.html"},
        external=True,
    )


@nox.session(venv_backend="none")
def package(session: nox.Session) -> None:
    """
    Check the crate packages and builds from exactly what crates.io would get.
    """
    session.run("cargo", "package", "--locked", *session.posargs, external=True)


@nox.session
def fixtures(session: nox.Session) -> None:
    """
    Regenerate the differential-testing fixture corpus from the C++ oracle.

    Requires `uv` on PATH; see scripts/gen_fixtures.py.
    """
    session.run("python", "scripts/gen_fixtures.py", *session.posargs, external=True)
