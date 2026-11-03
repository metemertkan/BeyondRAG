# Episode 2 fixtures

Synthetic PDFs for structure-aware parsing and soft versioning acceptance tests. No secrets; finance numbers are invented for demos.

| File | Purpose |
| --- | --- |
| `pdfs/policy_v1_liquidity_table.pdf` | Finance table (liquidity / FX caps). Marker cells: `HKD 450,000,000`, `USD 10,000,000`, `HKD 85,000,000`. |
| `pdfs/policy_v2_liquidity_table.pdf` | Same logical policy, revised bytes. Marker: `HKD 520,000,000`, LCR `110%`. Use same `source_key` as v1. |
| `pdfs/ops_compliance_multicolumn.pdf` | Two-column layout. Markers: `TOKEN_LEFT_COLUMN_ALPHA`, `TOKEN_RIGHT_COLUMN_BETA`. |
| `pdfs/policy_mispaired_limits_table.pdf` | Column-major painted limits table. Flat PyPDF clusters labels then values (neighbor after `Intraday cash buffer` is another label). Layout-aware table rows keep `Intraday cash buffer` ↔ `HKD 450,000,000`. Use for Short before/after — not v1/v2. |

Regenerate:

```bash
python3 scripts/generate_ep2_fixtures.py
```
