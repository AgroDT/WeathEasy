# Testing instructions for agents

Follow these instructions when writing, refactoring, or auditing WeathEasy tests.
Keep tests that detect plausible defects in our code or its integration with
dependencies. Test count, assertion count, and coverage are not measures of
protective value.

## Decide what deserves a test

Before adding or retaining a test, identify:

- The behavior or invariant owned by our application.
- A plausible defect that the test must detect.
- Why that defect would change the assertion.
- An independent source for the expected result.

If these cannot be identified, remove the test or redesign it. Do not add
ceremonial comments listing these points to every test.

Prioritize numerical correctness, unit conversions, coordinates and dates,
missing-data handling, publication of completed results, preservation of data
after failures, retry and restart behavior, resource ownership, cancellation,
and timeouts. Preserve tests of validation implemented by our own code.

Remove redundant cases when another test protects the same behavior at the
appropriate boundary. Do not remove a meaningful test merely because it uses
several assertions or exercises a dependency.

## Respect ownership of behavior and contracts

Test our units and their integration. Do not test the implementation of argparse,
Pydantic, PyArrow, xarray, Zarr, HTTP clients, or other dependencies.

Do not add standalone tests that merely confirm a dependency's public contract
or a data provider's documented API. For example, a test that only expects
PyArrow to reject a missing column or Pydantic to reject an extra field adds
little value. Let incompatible external contracts fail honestly at runtime;
do not add fallback behavior or validation solely to make such tests pass.

This does not exclude our own validation or processing rules. Preserve tests
of explicit checks implemented by our code, such as incompatible grids,
invalid intervals, or incomplete artifacts. Test our use of provider data:
field selection, conversions, aggregation, timestamps, coordinates, and NaNs.
Use documented semantics to derive expectations, without copying an entire
provider catalogue into test assertions.

A malformed external response is useful test input when the test checks our
response to failure. Preserve tests that prove raw inputs survive, incomplete
results remain unpublished, committed state is preserved, or retries recover.
A check that an injected exception simply propagates, without another relevant
effect, is a candidate for removal.

Reading an artifact written by our code through a real dependency can be a
valuable integration test. Verify known values and meaningful metadata; a
writer and reader agreeing with each other is insufficient evidence by itself.

## Keep unit tests focused

Each unit test should check one behavior or hypothesis. Split independent
calculations, validation failures, policies, and boundary cases into separate
tests. Do not combine them just because they share setup.

Several assertions are appropriate when they describe the same result or
event. For example, cancellation reaching I/O and releasing its read slot
belong to one scenario. A single dictionary assertion can still hide several
independent hypotheses; count behaviors rather than assertions.

Keep coherent integration sequences together when the sequence is the behavior
under test, such as failure followed by retry and recovery. Split independent
failure modes, such as an invalid provider response and an interrupted storage
write. Keep end-to-end workflows that verify our application behavior.

## Use pytest fixtures deliberately

Place all fixtures at the beginning of each test module, after imports,
constants, and type-checking imports, and before helper functions, classes,
and tests.

Name fixture functions as actions, such as `create_`, `init_`, or `get_`. Set
the injected result name explicitly with `name`:

```python
@pytest.fixture(name='weather_store')
def create_weather_store(tmp_path: Path):
    ...
```

Move repeated preparation into local fixtures. Keep the action under test and
its assertions in the test. Express fixture dependencies through arguments;
prefer a parameterized fixture over selecting fixtures by string through
`request.getfixturevalue`. Construct only the requested variant.

Use `pytest.mark.usefixtures` for shared preparation that returns no result.
Keep `autouse` limited to module-wide infrastructure needs, such as isolating
environment variables. Do not hide business-specific preparation in it.

Use function scope for mutable stores, models, arrays, and counters. Consider
broader scopes only for safely reusable data or resources, after measuring
setup costs. Do not share mutable state between tests to save time.

Use `yield` when a fixture owns a resource that must remain available during
the test and be closed afterward. Prefer a context manager around `yield` so
cleanup also runs when the test fails. Use `return` for fixtures that need no
teardown. Built-in `tmp_path` and `monkeypatch` already manage their own
lifecycles.

Keep ordinary helpers when they make setup clearer; do not convert every
helper into a fixture or introduce a large shared fixture framework.

## Parameterize equivalent scenarios

Use `pytest.mark.parametrize` or fixture `params` for cases of the same
behavior: sources, storage backends, boundary values, and failure states.
Split tests when the cases represent different hypotheses.

Give complex cases descriptive `ids`, such as `expired-job-resubmitted` or
`operation-timeout`. Automatic IDs are sufficient for clear string or numeric
values. Avoid opaque tuple and object IDs. Use a Cartesian product only when
every combination is meaningful.

## Isolate external effects and simulated failures

Use controlled fakes or monkeypatches for external I/O, nondeterminism, and
failure injection. Preserve the response structure and effects our unit relies
on. Assert observable results or meaningful interaction arguments; do not
merely prove that a mock returned its configured value or was called.

Use `monkeypatch.context()` for temporary failures. Limit the patch to the
operation being interrupted, then leave the context before testing recovery.
Do not manually save and restore functions just to undo a patch. Retain a
reference to the original function when a wrapper must delegate to it.

Activate failure injection after unrelated setup has finished. A simulated
provider-file write failure must not accidentally fail cache initialization.
Confirm that the setup reaches the intended branch. Assertions in callbacks
are meaningful only when the callback actually runs.

## Keep expected results independent

Prefer hand-derived examples, explicit specifications, published numerical
examples, or simpler independent calculations. Use invariants when exact
values are impractical.

Do not calculate expected results through the function under test or a helper
that repeats its algorithm. Agreement between training and application can be
an integration invariant, but also protect the underlying calculations with
independently derived expectations.

For numerical results, choose explicit tolerances appropriate to the behavior.
Check NaNs and missing dates where they carry meaning. Do not update expected
values, weaken assertions, or add skips merely to make a failing test pass.

For important regression tests, identify a small semantic change that the test
would detect. When safe and inexpensive, verify this with a temporary mutation
in an isolated copy or safely restorable worktree. Restore the original code
immediately. Mutation testing is not mandatory for mechanical refactors,
framework wiring, or expensive integration scenarios.

## Test application behavior through the CLI

Keep CLI tests that verify dispatch, argument propagation, storage selection,
CSV data, generated artifacts, or application outcomes.

Do not test help wording, default-path formatting in help, argparse's standard
rejection of unsupported flags, or progress-bar internals. A small test proving
that metadata commands avoid optional imports protects our import structure;
it should not also snapshot help output or enumerate every parser variant.

## Add plugins only for a concrete need

Keep controlled fakes for failures, retries, and restarts. Do not introduce
cassette testing as a general replacement. A cassette replays an old response
and cannot detect changes in a live provider contract. Consider it only for
specific tests of our request construction and transport integration, with
compatible clients, filtered credentials, and deliberate recording and replay
settings. Keep large GRIB and NetCDF inputs as separate file fixtures.

Consider `pytest-regressions` only for large, independently validated numerical
or file outputs that are cumbersome to express directly. Keep compact explicit
assertions for small examples. Do not add snapshots of help, OpenAPI, complete
Zarr metadata, or binary archives merely to freeze current output.

Review baseline changes as changes to expected behavior. Automatic regeneration
does not establish correctness. Compare decoded values and meaningful metadata
rather than serialization details, and specify numerical tolerances.

## Verify changes and report evidence

Run focused tests first, then the relevant broader suite. For a full run, use
the declared optional dependencies:

```sh
uv run --all-extras --frozen pytest
```

Inspect the skip summary. `importorskip` is appropriate for intentionally
optional components, but a passing run that skips their tests is not evidence
that those integrations work. Ensure the main CI test environment includes
the required extras. Helm must be available for chart tests.

Use `--durations=20` to measure slow tests before changing fixture scopes or
adding performance infrastructure. Use the repository's Ruff checks and
`git diff --check` for test edits. Documentation-only edits require document
and diff checks, not an application test run.

Report what changed, what behavior remains protected, which checks ran, and
any skipped or environment-blocked checks. Distinguish code execution from
behavioral protection. Preserve unrelated staged and untracked work.
