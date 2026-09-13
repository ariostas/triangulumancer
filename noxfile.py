from __future__ import annotations

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
def fixtures(session: nox.Session) -> None:
    """
    Regenerate the differential-testing fixture corpus from the C++ oracle.

    Requires `uv` on PATH; see scripts/gen_fixtures.py.
    """
    session.run("python", "scripts/gen_fixtures.py", *session.posargs, external=True)
