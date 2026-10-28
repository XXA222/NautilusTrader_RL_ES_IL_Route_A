# F05 — Source access and unconsumed Train audit

Status: `SOURCE_SCOPE_REGISTERED_ACQUISITION_PENDING`.

## Read-only ledger findings

`.artifacts/research-access.sqlite` was queried read-only. It has 713 access records: 708 `development/recorded`, 3 `test/reserved`, and 2 `holdout/reserved`; `query_intent`, `trial_attempt`, and `trial_attempt_event` are empty. The latest development coverage ends on 2026-10-02. There is no later registered development access or formal query intent to authorize another candidate experiment.

The 2026-08-12—2026-10-02 Train snapshot appears in the ledger under many development identities, but the existing physical-source-scope audit reports its monthly funding archive overlaps protected Test/Holdout source coverage. These prior reads therefore do not provide a clean, unused source. The 2026-10-02—2026-10-22 candidate source remains incomplete because its funding data was unavailable; existing records document that it was not used for a valid complete market-source study. Prior July/September monthly funding imports also have documented physical-source overlap and are ineligible.

This audit did not open or decode any market source file, monthly archive, Test window, or Holdout window. It does not infer source legality from a clipped scoring interval when the imported physical archive covers a wider protected period.

## Decision

No new, unconsumed, physically non-overlapping, complete Train source is evidenced by the current ledger and local source-audit records. F06 economic selection/replay, F08 value-model research, and F09 qualification/student progression must not proceed on the basis of consumed or invalidated windows. F07 source-independent unit engineering is not blocked by this finding. Continue economic research only after a newly authorized Train source is registered before read and its full physical source coverage is verified against reserved Test/Holdout coverage.


## 2026-10-26 source-path re-audit

The ledger was rechecked read-only. A proposed 2026-10-03—2026-10-23 Train-inner interval has no overlap with registered Test/Holdout records or formal query intents. This calendar check alone does not make the interval eligible: the earlier `routea_teacher_train_20261002_20261022` attempt had already fetched daily ETHUSDT klines, mark-price klines, and index-price klines for 2026-10-02—2026-10-22 before the ledger contained a matching development reservation. Its import is incomplete: `IMPORT_FAILED.json` records a funding-history request timeout after 25 seconds. The 2026-10-25 preregistration file does not cure the missing pre-read ledger registration. The already-read partial source is therefore not treated as a new, unconsumed Train source, and no native economic run may use it.

The importer has a bounded USDⓈ-M funding API fallback for the current month, and the official endpoint documents inclusive `startTime`/`endTime` millisecond bounds with a 1,000-record maximum. This establishes a possible future acquisition route, not successful or complete source availability. Binance's official public-data README says daily archives are published the following day, while monthly files arrive on the first Monday; that publication cadence does not resolve the observed API timeout. No source request, archive, or market event was opened during this re-audit.

**At the time of this audit, the decision was `BLOCKED_BY_SOURCE`.** The next section records the subsequent preregistered source scope and changes the current state.


### Local import-manifest gap scan

A read-only scan of all 15 `data/*/import.json` manifests compared each imported scope, archive URL granularity, and source span against the ETHUSDT development/test/holdout ledger. It did not read `market.parquet`, raw ZIP contents, or market values. Among complete dated imports whose physical source spans do not overlap registered Test/Holdout, every available interval overlaps recorded development access; no clean unused contiguous gap of seven days or longer was found. Several snapshot manifests lack direct date fields and were not treated as eligible without a source-specific reconstruction. This closes the local-manifest alternative for a multi-window Train screen; it does not claim anything about future public data availability.


## 2026-10-26 registered Train scope

Following the user's instruction to continue, a new development scope was frozen before market-source read and recorded in `.artifacts/research-access.sqlite`:

- Protocol: `route-a-teacher-student-train-20261023-20261130-v1`.
- Source: `[2026-10-23T00:00:00Z, 2026-11-30T00:00:00Z)`; scored window begins 2026-10-25 after two warmup days.
- Registry key: `access:route-a-teacher-student-train-20261023-20261130-v1`; prereg SHA-256 `e7f795fce9f8539e58085f6bd85449f96edb53324cebcbb6380dec47c8d38aca`; state `development/recorded`.
- Teacher screen: nine frozen T01–T08 X1 configurations (T01 old and reversed lifecycle) in four 5-day nonoverlapping windows, Base/Stress and two seeds, against Flat and fixed-long/fixed-short controls.
- Student fit, only if one teacher passes: teacher labels and DAgger collection restricted to the teacher-screen windows. Student evaluation uses a separate four-window late Train holdout; no training/DAgger collection on those windows.
- Funding acquisition is frozen as exact date-bounded USD-M REST requests for October 23–31 and November 1–29. This avoids opening the October monthly funding ZIP, whose physical source month overlaps earlier development access. Importer changes bind every physical source span to the matching pre-read registry entry and reject prior development or Test/Holdout overlap.

The teacher IDs were corrected in a pre-read erratum after checking the code registry; a second pre-read source erratum freezes the funding granularity and importer hash. These errata supersede the parent protocol without changing the dates, candidate acceptance rule, or outcomes (none have been observed). Current status is **`SOURCE_SCOPE_REGISTERED_ACQUISITION_PENDING`**. No market source or event data has been read under this new registration yet; native teacher/student runs remain pending the complete source and the evaluation runner.

## 2026-10-27 source-independent execution preparation

The read-only preflight confirmed the registered access row is still `recorded`, the source range remains `[2026-10-23T00:00:00Z, 2026-11-30T00:00:00Z)`, and there is no imported dataset at the frozen output path. It returned `PREFLIGHT_PASS_SOURCE_PENDING` with `market_payload_read: false`.

Before any source read, the preflight found that four teacher implementation files no longer matched the hashes in the parent preregistration. The source erratum was amended in place to bind the tested implementation bytes, the execution template, and the new native screen runner. The candidate matrix, source dates, controls, and acceptance rules did not change. It also freezes the scaler fit to the two unscored warmup days, 2026-10-23 through 2026-10-25. The amended source erratum remains chained to the already registered parent protocol; the access row and market-data scope were not changed.

`qa/run_route_a_teacher_student_train_screen_20261026.py` executes one complete teacher window at a time. Active-goal execution covers six configurations: EMA exit-only, EMA reverse-after-flat, legacy ROC, the pre-frozen F04 cost-aware/hysteresis ROC X1 candidate, Donchian, and Z-score. The runner excludes T05–T08, records declared/evaluated/additional/excluded IDs in its summary, and keeps the per-candidate acceptance rule unchanged. The cost-aware ROC receives only the registered Base/Stress round-trip cost; no threshold is retuned. The runner pairs these candidates with Flat/fixed-long/fixed-short under Base/Stress and both registered seeds, verifies native M2.2 barriers, and reconciles each run to the same protocol/window/scenario/seed. Student training remains ineligible unless an in-scope teacher passes the complete screen. Ten focused runner tests, Ruff, and targeted MyPy pass. The first five-day scored window ends 2026-10-30; no partial window was scored.

Old S06 economics were not rerun after this audit identified source overlap/consumption constraints. Their prior negative results remain development history, not evidence for the newly registered Train scope.

## 2026-10-27 staged warmup acquisition

The first registered, source-bounded acquisition is now complete for the two warmup dates only: `2026-10-23` through `2026-10-24`. Six Binance daily ZIPs were checksum-verified (klines, mark price, and index price for both days). The direct, exact-date `/fapi/v1/fundingRate` transport timed out; the same USD-M request was captured through Binance's read-only funding-history connector. Its canonical six-row payload SHA-256 is `f2491249c66b3a531902ab335eaa40ba2d3f9345f49000faedb8bbad2c4bab84`, with a separate receipt documenting the connector capture. The failed direct request remains in `data/routea_teacher_warmup_20261023_20261024/IMPORT_FAILED.json`; the successful retry reused the verified daily-archive cache.

The curated warmup manifest contains 8,649 events, with bar coverage `[2026-10-23T00:01:00Z, 2026-10-25T00:00:00Z)`. A registered Train read produced a DQ-B scaler over `[2026-10-23T02:02:00Z, 2026-10-25T00:00:00Z)`: 2,756 feature rows, scaler hash `7e8cb0389b20f3f3fbe5cbd8b89298cb7a7a6645fff63e9658143625c7cc46da`, only `conservative_availability` as a quality finding, and `scored_data_used_for_fit: false`. The earlier v5 scaler attempts failed before any economic run: first on a row-group scope boundary, then because the two unscored days cannot supply the two daily aggregates required by its one-day-return feature. The source erratum records the resulting v4 feature-profile correction, which no frozen teacher's required input fields use; teacher windows, candidates, costs, and acceptance thresholds remain unchanged. The erratum also binds the exact funding capture method, source and runner code hashes, and the failed transport receipt.

Current read-only preflight passes with the parent Train registration still `recorded`. One of five staged teacher/warmup imports is present. The preflight itself did not decode the market payload (`market_payload_read: false`); the successful registered warmup read and scaler fit above did. No candidate result or economic outcome has been observed, no teacher window has been scored, and student training has not started. The combined 77-test targeted suite, Ruff, and targeted MyPy all pass after the importer, staged runner, and funding-calendar changes.

The first complete teacher window is `[2026-10-25T00:00Z, 2026-10-30T00:00Z)`. Its daily sources through October 29 are not all published yet, so no partial score is valid. On or after publication of the October 29 archives, import the frozen `teacher_window_imports[0]` span (`2026-10-23` through `2026-10-29`) to `data/routea_teacher_train_window0_20261023_20261029`, then run `.venv-ubuntu/bin/python qa/run_route_a_teacher_student_train_screen_20261026.py --window-index 0`. The funding cache for this larger exact date range must be captured under the same frozen connector method if direct REST remains unavailable. Continue with each later window only after all of its frozen source dates are published; never label a partial window complete.

## 2026-10-27 active-goal scope and hash repair

The first read-only preflight after the source-independent S06 work found stale frozen code hashes for `exits.py`, `registry.py`, and the native runner. Their current tested bytes were rebound in the existing source erratum before any scored-window outcome; no teacher threshold, source date, cost, seed, control, or per-candidate acceptance rule changed. The source erratum SHA-256 is now `2e0f0502130adef14d39e88ed69164234b4a2433ca7e3b090aadc09603a6ee39`. Read-only preflight now passes with registry state `recorded`, one of five staged imports present (the warmup only), and `market_payload_read: false`.

The active execution scope is T01–T04 plus the already frozen F04 cost-aware ROC candidate; T05–T08 remain excluded from native evaluation. The summary distinguishes completion of this scope from completion of the original nine-row preregistered matrix. The registered window-0 import manifest is not present yet, and its five-day scoring window is not complete. Do not score partial bars or launch student training. The focused runner suite passes 10 tests, Ruff and MyPy pass, and the runner records its in-scope, additional, and excluded candidate IDs in each summary.

## 2026-10-28 pre-score status binding

Before any score exists, the native screen runner was tightened to require each run receipt to state `economic_status: SCENARIO_ONLY` and `qualification: D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2` during both resume and aggregation; the summary now repeats that qualification at top level. This prevents a mislabeled run receipt from entering the frozen screen. Candidate recipes, source dates, costs, seeds, controls, and acceptance rules are unchanged. Runner hash `07a6aa9d4eabddf71bf43f48921310e791d97c06767e2a488887193bb21527ce` is bound in the existing source erratum, whose SHA-256 is now `bb72b06db2ab02242c1c859d561f9cd0486ebad966b6e783dd94ca219f4b07b5`. Read-only preflight passes with one of five staged imports present and `market_payload_read: false`. Targeted runner tests (17), Ruff, and focused MyPy pass. No scored window was run.


## 2026-10-28 EMA rejected-close lifecycle repair

The EMA reverse-after-flat state machine previously cleared the one-shot reverse candidate when a native close order was rejected in the synthetic lifecycle test. Because EMA crosses are edge-triggered, later bars then had no exit signal and could hold the original side indefinitely. It now retains the candidate, removes only rejected/canceled/expired order IDs, and requests a close retry on the next legal 1m decision. A reverse entry remains gated by the existing native full-fill proof, flat account, no pending orders, and a fresh direction check. The synthetic regression proves a rejected close, retry, confirmed fill, one reverse entry, and no duplicate same-bar entry. Long-to-short and short-to-long tests remain in the focused lifecycle suite.

Source erratum binds the updated base implementation SHA-256 `e90fb5c1b2cd4f6523483979308c75140deb06bb533e433379baad1c50f93393`; source-erratum SHA-256 is `e5ec4643fa7dba1f289d783bbcf9809a9ffaefb6533073b3c7d8085685fc63f9`. The preregistered recipe, window roles, costs, controls, seeds, and acceptance criteria are unchanged. No market payload or economic outcome was read for this repair.


## 2026-10-28 ATR exit on first fresh bar after a gap

The gap handler returned MISSING_BAR before evaluating an ATR exit, even when the current observation contained a fresh complete bar. A close-through of the retained ATR stop on that bar could therefore be missed if price recovered on the next bar. X2 (and the shared ATR exit path) now evaluates the unchanged stop on that first fresh bar; indicator continuity still resets, and missing stop inputs while positioned still raise. Two synthetic regressions cover long and short gap-bar stop breaches.

Updated base implementation SHA-256 d70bb785ae38904f46e0f8eb5ce74de6fe7c83a53f73f13c6bac9a5fd53f3159; source-erratum SHA-256 9606980f82c5af81a1063163c808fec650350be082e2490fdac8ebfbe3136d3e. Candidate recipes and all frozen economics remain unchanged; no market payload or outcome was read.

## 2026-10-28 registered window-0 partial acquisition

The frozen importer was invoked for the registered window-0 source, `2026-10-23` through `2026-10-29`, with the preregistered 1 ms availability delay, access key, protocol SHA-256, daily archive granularity, and exact October funding API month. It checksum-verified and parsed the 12 official ETHUSDT USD-M daily archives for klines, mark-price klines, and index-price klines covering October 23 to 26. Each cached archive has its source URL and SHA-256 in the adjacent `raw/*.source.json` receipt.

The next request failed with HTTP 404. From the importer's date/stream iteration order and the successful receipts ending on October 26, the missing request is inferred to be the October 27 klines checksum. The importer wrote `data/routea_teacher_train_window0_20261023_20261029/IMPORT_FAILED.json`; it created neither `import.json` nor `market.parquet`. The funding API stage was not reached. No partial source was scored, no native economic result was produced, and no Test/Holdout source was requested or read.

This attempt did read market payloads from the registered Train scope through October 26. The subsequent metadata-only validator reports `market_payload_read: false` because that validator did not decode payloads; that field does not describe the preceding importer run. Its current result is `PREFLIGHT_PASS_STAGED_SOURCES_PARTIAL_METADATA_NOT_READ`, with 2 of 5 staged paths present. The second path is raw-only and must not be treated as a completed import; the scoring runner still requires the verified manifest, completed parquet receipt, and full source coverage.

After the remaining registered daily archives are published, resume the same immutable output (the importer permits an `IMPORT_FAILED.json` raw-cache-only directory) with the frozen command:

```bash
.venv-ubuntu/bin/python -m rles_data.binance_archive --symbol ETHUSDT --start-day 2026-10-23 --end-day 2026-10-29 --output data/routea_teacher_train_window0_20261023_20261029 --delay-ns 1000000 --funding-api-month 2026-10 --development-access-key access:route-a-teacher-student-train-20261023-20261130-v1 --development-protocol-sha256 e7f795fce9f8539e58085f6bd85449f96edb53324cebcbb6380dec47c8d38aca
```

Do not score window 0 until the complete daily source range, funding input, import manifest, and frozen preflight all pass.

## 2026-10-28 importer failure receipt source attribution

The importer now includes `source_url` and `http_status` in `IMPORT_FAILED.json` when the exception provides them; the existing exception type and detail remain. Both daily and range import paths use the same receipt builder. A synthetic HTTP 404 regression verifies the captured URL and status. This test made no network request and read no market data.

The frozen importer binding was advanced from `07a482a9d00d306cab66b7cca4f9c74a17b7f1ef26c4ed925d1c967c233ed056` to `06ce56358ca04bb4b22209e99328b6149a234ca67302b97ee090fca072ba2259` in the existing source erratum before any later source acquisition. The earlier warmup import and partial window-0 acquisition used the previous importer bytes; this code change only affects failure-receipt serialization, so it does not change archive parsing or successful market payloads. Existing source caches were not reprocessed. The current source-erratum sidecar SHA-256 is `96f014804a462bb84a6386af1ba6e79432be839c3544969422a912ad81cff6ba`.

The preregistered date scope, recipe, costs, evaluation rules, and source bytes are unchanged. The targeted importer tests passed (2 passed, 27 deselected), and Ruff passed for the importer and test file. No economic scoring or Test/Holdout access occurred.
