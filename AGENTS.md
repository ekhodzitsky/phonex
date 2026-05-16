# phonex Agent Guide

This repository uses COAD: https://github.com/ekhodzitsky/coad

Before editing, identify the smallest relevant workcell and read its
`MODULE_CONTRACT.md`, `README.md`, and `TODO.md`. Keep writes inside the
declared ownership boundary unless the contract says to escalate.

Run the workcell verification commands for the module you changed, then run:

```bash
coad check .
```

`coad check .` validates methodology compliance. It does not replace the Rust
build, tests, or manual review needed for behavior changes.
