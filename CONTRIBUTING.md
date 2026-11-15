# Contributing

Issues and pull requests are welcome. For anything larger than a small fix,
please open an issue first so the approach can be discussed.

## Setup

You need a Rust toolchain (1.86 or newer for the tests and benchmarks, though the
library itself builds with 1.85; via [rustup](https://rustup.rs/)),
[uv](https://docs.astral.sh/uv/), and [prek](https://github.com/j178/prek) (or
`pre-commit`) for the linters. There are no system libraries to install.

```bash
git clone https://github.com/ariostas/triangulumancer.git
cd triangulumancer

cargo test
uv venv .venv-dev
VIRTUAL_ENV=.venv-dev uv pip install -e '.[test]'
.venv-dev/bin/python -m pytest
```

The editable install compiles the extension module through maturin. Rerun the
`uv pip install` after changing Rust code, or the tests run the old build.

## Checks

Pull requests run these in CI, on Linux, macOS and Windows:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --features python --all-targets -- -D warnings
cargo test
.venv-dev/bin/python -m pytest
prek -a --quiet                     # ruff, mypy, codespell, and more
```

`cargo test` turns on the internal `self-check` feature (through the crate's
dev-dependency on itself), which in debug builds re-checks the regular
triangulation after every insertion and every incremental flip set against a
naive recomputation. Both are quadratic, so they stay out of dependents' builds.

[nox](https://nox.thea.codes/) has sessions for a few more:

```bash
nox -s readme     # the README's examples, as doctests
nox -s docs       # the Rust documentation, as docs.rs builds it
nox -s package    # build the crate from exactly what crates.io would get
```

## Layout

The algorithms are plain Rust in `src/`; `src/python.rs` holds the PyO3 bindings,
which only convert types and contain no logic. The combinatorial algorithms take
an oriented matroid, never a configuration, and that boundary is deliberate.
[`AGENTS.md`](AGENTS.md) describes the conventions, and [`PLAN.md`](PLAN.md) the
design: [`CGAL.md`](CGAL.md) and [`TOPCOM.md`](TOPCOM.md) document the
algorithms the implementation follows.

Two behaviours are relied on by the tests and by users, and must not change by
accident:

* Output order is canonical: vertices sorted within each simplex, simplices
  sorted within each triangulation, and the enumerations' lists sorted.
* Errors are values of `triangulumancer::Error`, mapped to Python exceptions.
  Nothing panics on bad input.

## Tests against the 0.3 implementation

`tests/fixtures/` holds the output of the C++ implementation (`triangulumancer`
0.3.0) and of TOPCOM's command-line tools over a corpus of configurations, and
the Rust tests compare against it. Some of the 0.3 results are known to be wrong;
read [`tests/fixtures/KNOWN_DIVERGENT.md`](tests/fixtures/KNOWN_DIVERGENT.md)
before writing tests against the corpus. Regenerating it is rarely needed and is
described in `scripts/gen_fixtures.py` and `scripts/gen_topcom_fixtures.py`.

## Commits

Use [conventional commits](https://www.conventionalcommits.org/), e.g.
`fix: reject subnormal weights`. Pull request titles are checked for the same
format.
