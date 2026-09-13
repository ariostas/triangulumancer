# Working in this repository

## What this is

`triangulumancer` computes triangulations of point and vector configurations.
It is being rewritten from C++ (CGAL + TOPCOM behind pybind11) to pure Rust with
PyO3 bindings. **Read `PLAN.md` before making changes** — it defines the phases,
the architecture, and the decisions that have already been made.

The two companion documents are design input, produced by reading the upstream
sources, and should be treated as the reference for the algorithms:

* `CGAL.md` — regular and Delaunay triangulations.
* `TOPCOM.md` — placing and fine triangulations, flips, and the two enumerations.

## Layout

```
src/            the Rust crate; algorithms are plain Rust, bindings are feature-gated
  python.rs     PyO3 translation layer only — no logic here
python/         the Python package and its tests
scripts/        development tooling (fixture generation)
tests/fixtures/ reference outputs from the C++ oracle
```

The layering in `src/` is load-bearing (`PLAN.md` section 6): combinatorial
algorithms take an oriented matroid, never a configuration. Do not reach past
that boundary for convenience — it is what keeps the rest of TOPCOM's
functionality cheap to add later.

## Commands

```bash
cargo test                                    # Rust tests
cargo clippy --all-targets -- -D warnings     # and again with --features python
cargo fmt
prek -a --quiet                               # all linters (not `pre-commit run -a`)

uv venv .venv-dev && VIRTUAL_ENV=.venv-dev uv pip install -e '.[test]'
.venv-dev/bin/python -m pytest                # Python tests

python scripts/gen_fixtures.py --list         # inspect the fixture corpus
python scripts/gen_fixtures.py                # regenerate it (needs network + uv)
```

## Conventions

* Rust is the implementation language. `python.rs` converts types and nothing else.
* Every error is a variant of `crate::Error`, mapped to a Python exception at the
  boundary. Nothing panics on bad user input, and nothing calls `process::exit`.
* Public enums are `#[non_exhaustive]`.
* Output ordering is canonical and deterministic: simplices sorted internally,
  then sorted between themselves. Tests depend on this.
* Methods for phases that have not landed raise `NotImplementedError` via the
  `todo()` helper in `src/python.rs`. `python/tests/conftest.py` turns that into
  a skip, so the pytest suite lights up phase by phase with no test edits.

## Testing against the C++ oracle

The reference implementation is the published `triangulumancer==0.3.0` wheel, not
anything in this tree. `scripts/gen_fixtures.py` installs it into a throwaway
virtualenv and records the output of every public method over a corpus of
configurations.

The oracle has known bugs that are deliberately not fixed (`PLAN.md` section 3).
Affected results are marked `known_divergent` in the fixture data and must be
asserted to *differ* from the Rust implementation, never to match. See
`tests/fixtures/KNOWN_DIVERGENT.md`.

## Commits

Conventional commits, with an `Assisted-by:` trailer when applicable. No
`Co-authored-by`.
