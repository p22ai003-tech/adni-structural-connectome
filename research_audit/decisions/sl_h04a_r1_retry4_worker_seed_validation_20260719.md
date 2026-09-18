# Retry4 parent/worker seed-validation policy

Snakemake reloads the workflow in local worker subprocesses. Re-hashing the
entire 51-GiB immutable seed in every worker is redundant and can dominate the
runtime without improving the execution boundary.

Retry4 therefore applies two validation levels:

- the launcher fully re-hashes every declared seed file before execution;
- the parent Snakemake DAG process fully re-hashes every declared seed file;
- worker subprocesses re-hash the signed seed-manifest file and all small
  source evidence, enforce the exact `00_inputs`/`01_dwi` path boundary, and
  verify every declared file exists with its exact declared size.

Workers may use the reduced check only when their command line contains both
Snakemake `--mode subprocess` and `--target-jobs`. Any direct launcher, dry run,
or parent DAG parse performs the full content re-hash. The scientific workflow,
allowed rules, inputs, parameters, units, and resource limits are unchanged.
