# FreshCart Pipelines

End-to-end Data Engineering assignment covering Git and CI/CD,
FreshCart ELT on Snowflake, Mage ML pipelines, and RabbitMQ streaming.

## Project Structure

- elt/ - FreshCart ELT pipeline
- iris_project/ - Mage Iris classification pipeline
- stream/ - RabbitMQ dispatcher and drivers
- tests/ - Automated tests
- .github/workflows/ - GitHub Actions CI

## FreshCart ELT (Part B)

The Snowflake pipeline is orchestrated by `elt.run_pipeline`. Configure the
Snowflake variables in `.env` using `.env.example` as a template. Run the
baseline with mock source data:

```powershell
python -m elt.run_pipeline --mode mock
```

Use `--mode transform-only` (the default) to rerun transformations against
already-loaded Bronze data without appending another extraction batch:

```powershell
python -m elt.run_pipeline --mode transform-only
```

Run an individual transformation stage with `--stage`, for example:

```powershell
python -m elt.run_pipeline --stage gold
```

Pincode responses use deterministic mock data unless `--live-pincodes` is
specified. SQL transformations are in `elt/sql/`; pure-Python rule tests are
run with `python -m pytest`. Live extraction requires the service URLs and
tokens documented as variable names in `.env.example`.

Bronze file ingestion is tracked in the additive
`elt/sql/01_ingestion_files.sql` migration. The ledger identity is SHA-256 of
the normalized source plus staged `source/date/filename` path; the compressed
file's content checksum prevents a changed file from silently reusing that
identity. A `LOADING` claim is written before PUT/COPY and becomes `LOADED`
only after COPY results and Bronze source-file row counts are verified. Failed
attempts are marked `FAILED` and can be retried. A process crash can leave a
`LOADING` entry; inspect its Bronze `SRC_FILE` row count and verify no loader is
still active before manually reconciling its ledger status. The ledger is a
standard Snowflake table for trial-account compatibility. Its `FILE_ID` has no
enforced uniqueness constraint: ingestion is single-writer only. A host-level
lock prevents competing local processes, but all scheduled jobs and manual
loaders must also be coordinated so that only one writer runs across hosts.
The loader checks for duplicate ledger rows and refuses ambiguous legacy Bronze
rows when no ledger entry exists; reconcile those rows before retrying.
Snowflake PUT, COPY, and ledger DML are separate operations, not one atomic
transaction, so failed or crash-interrupted files require the documented
row-count/status reconciliation before retry.

## Assignment Components

- Part A: Git and CI/CD
- Part B: FreshCart ELT on Snowflake
- Part C: ML Pipeline with Mage
- Part D: Food-order streaming with RabbitMQ
- Part E: Final report and viva
