# Project cross-platform rules

Changes that touch `os`, `pathlib`, `tempfile`, `shutil`, `subprocess`,
`multiprocessing`, `zipfile`, native handles, encodings, links, atomic replace,
or `fsync` must follow the cross-platform contract in `CONTRIBUTING.md`.

- Audit the complete lifecycle, not only the API named by the first failure.
  For example, review archive creation, file flush, publication, validation,
  cleanup, and rejection reporting together.
- Flush regular files only through write-capable handles. Directory `fsync` is
  POSIX-only; Windows durable replacement must use the project's declared
  Windows primitive.
- Subprocess text capture must explicitly use UTF-8 with strict decoding.
  Do not rely on the runner's console code page.
- Do not use `os.kill(pid, 0)` as a Windows liveness probe. Normalize only the
  documented pipe-closure errors and propagate unrelated operating-system
  errors.
- Fail-closed paths must also fail visibly: retain the stage and sanitized
  exception type/message instead of silently returning status 1.
- A reduced SDK proof does not cover a production orchestration lifecycle.
  Every platform repair needs a focused semantic regression and the implicated
  production path in the Ubuntu/Windows Architecture matrix.
- Do not report CI or tests as green until the process has a numeric exit code
  and final summary. Both Windows Python 3.10 and 3.13 Architecture jobs are
  required before approval.
