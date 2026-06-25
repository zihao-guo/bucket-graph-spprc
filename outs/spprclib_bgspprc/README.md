# spprclib bgspprc Results

Run directory: `20260625_200720`

CSV: `20260625_200720/bgspprc.csv`

Statistics use the same convention as `benchmarks/README.md`: shifted geometric mean with shift `1s`, arithmetic mean in seconds, and `120s` substituted for timeout/error rows.

## Reproduced Summary

| set | ng | solver | sgm (s) | mean (s) | solved |
|---|---:|---|---:|---:|---:|
| spprclib |  8 | bgspprc | 0.993 | 7.556 | 44/45 |
| spprclib | 16 | bgspprc | 2.273 | 16.248 | 40/45 |
| spprclib | 24 | bgspprc | 6.440 | 27.171 | 38/45 |

## Reference From Project README

| set | ng | solver | sgm (s) | mean (s) | solved |
|---|---:|---|---:|---:|---:|
| spprclib |  8 | bgspprc | 0.893 | 6.953 | 44/45 |
| spprclib | 16 | bgspprc | 2.000 | 15.674 | 40/45 |
| spprclib | 24 | bgspprc | 5.683 | 26.354 | 38/45 |

## Notes

- The solved counts match the README reference rows.
- Runtime is machine/load/compiler dependent; this run is slightly slower than the committed reference table.
- `ERROR(137)` rows are stored with empty cost/time fields in the CSV and counted like timeout rows for summary statistics.
