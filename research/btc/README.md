# BTC Research

This directory is the permanent bookkeeping area for BTC research in Oracle.

Rules:

- Do not overwrite prior champions or failed branches.
- Every meaningful experiment gets a frozen checkpoint entry.
- Every new strategy track gets a diary entry before it starts.
- Production sleeves must reference frozen checkpoints, not mutable scratch outputs.

Structure:

- `diary.md`: chronological BTC research log.
- `checkpoints/index.json`: append-only checkpoint ledger.
- `checkpoints/<checkpoint-id>/`: copied manifests and local notes for a frozen step.
- `projects/<project-id>/`: active strategy-track plans and local notes.

Current active next-track project:

- `projects/btc-multivenue-v1/`
