`probes_common.py` and `probes_long_context.py` are copied verbatim from
https://github.com/OmarMujahid/jev-decision-bench (MIT). They regenerate that benchmark's
needle-in-a-haystack tasks deterministically (seed 7, SQuAD v1.1 validation filler), so the
published per-item Jev results in `results/jev-decision-bench-needle/` can be rejoined against
needle depth. Run via `eval/jev_needle_by_depth.py`.
