# Contributing

## Cross-platform process and filesystem contract

This project supports Ubuntu and Windows. Platform-sensitive work is reviewed
as a complete lifecycle rather than as a one-line compatibility patch. A fix
for an archive operation, for example, must cover creation, file flush,
publication, validation, cleanup, and rejection reporting.

The following rules are normative:

1. Regular files are flushed only through write-capable descriptors or streams.
   Directory `fsync` is POSIX-only. Windows durable replacement uses the
   repository's `MoveFileExW(..., WRITE_THROUGH)` adapter where replacement is
   part of the durability boundary.
2. Process supervisors normalize only known endpoint-closure conditions from
   both `poll()` and `recv()`. Unrelated `OSError` subclasses propagate with the
   current phase. Windows liveness checks must use Win32 process APIs with exact
   ctypes declarations and closed handles; `os.kill(pid, 0)` is POSIX-only.
3. Every captured subprocess text stream declares `encoding="utf-8"` and
   `errors="strict"`. User-facing CLIs establish UTF-8 output before emitting
   non-ASCII help or error text.
4. Durable publication is explicit about replace/no-replace behavior. Cleanup
   removes only artifacts whose published identity is still owned; it never
   deletes a pathname based only on an earlier assumption.
5. Fail-closed operator commands preserve a structured rejection stage and a
   sanitized exception type/message for expected operational errors. Returning
   status 1 without a reason is not acceptable.
6. Performance assertions measure the promised boundary. Interpreter startup,
   imports, and runner scheduling are not part of a supervisor-detection SLO;
   runner-sensitive end-to-end wall bounds are recorded separately.

### Required evidence

Every platform repair must include all of the following:

- a focused regression that fails before the repair and passes afterward;
- a semantic test that simulates the other operating system when practical;
- execution of the real production path implicated by the failure;
- terminal numeric results from the Ubuntu/Windows Architecture matrix on
  Python 3.10 and 3.13.

The reduced Windows LanceDB job proves only create/reopen/query SDK behavior. It
does not replace the full archive, ledger, process, or cleanup lifecycle gates.
Intermediate log output, a live session identifier, and progress percentages
are not test verdicts.

`tests/test_portability_contract.py` enforces the static portions of this
contract. Behavior-specific regressions remain next to the production feature
tests so the Architecture matrix exercises them on both operating systems.
