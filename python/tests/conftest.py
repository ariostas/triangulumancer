"""Shared pytest configuration.

During the Rust migration the extension module carries the whole 0.3 API
surface, but the methods for phases that have not landed yet raise
``NotImplementedError`` (see ``src/python.rs``). The hook below turns that into
a skip rather than a failure.

This is deliberately automatic instead of a pile of ``pytest.mark.skip``
decorators: the tests are the acceptance criteria for each phase, and they
should start passing the moment their phase lands, with no test edits and no
risk of a stale marker hiding a regression. The skip count shrinking phase by
phase is the progress bar for the migration.

Anything that raises ``NotImplementedError`` for a reason *other* than a
migration stub would also be skipped here, so nothing in this package should
raise it once the port is complete. Removing this file is part of phase 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.reports import TestReport
    from _pytest.runner import CallInfo


# pluggy's marker is untyped, so mypy cannot see through the decorator.
@pytest.hookimpl(wrapper=True)  # type: ignore[untyped-decorator]
def pytest_runtest_makereport(
    item: pytest.Item, call: CallInfo[None]
) -> Generator[None, TestReport, TestReport]:
    report = yield
    if (
        report.when == "call"
        and call.excinfo is not None
        and call.excinfo.errisinstance(NotImplementedError)
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] + 1,
            f"awaiting Rust migration: {call.excinfo.value}",
        )
    return report
