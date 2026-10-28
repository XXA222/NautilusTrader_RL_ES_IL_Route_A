# F02 — Cost and churn diagnostics

Price gross is inferred per trade as net effect + fee + slippage - funding, an accounting identity rather than independently observed gross fill PnL. Controls have account-row reconciliation only. Source is existing consumed S07 native reports/trade JSONL/control account CSVs; no market source data read.

Reconciliation errors: 0; all 48 teacher and 114 control account-row checks included.

| Teacher | Exit | Scenario | Runs | Cases | Fees | Slippage | Funding | Net | Inferred gross | Mean run holding min | Gross <=0 | Positive <= costs | Gross > costs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T01_EMA_9_21 | X1 | base | 3 | 2534 | 629.360737 | 128.002970 | -4.551368 | -635.656585 | 126.258490 | 21.814 | 1684 | 354 | 496 |
| T01_EMA_9_21 | X1 | stress | 3 | 2534 | 1258.721532 | 631.405270 | -4.502226 | -1768.272618 | 126.356410 | 21.814 | 1683 | 559 | 292 |
| T01_EMA_9_21 | X2 | base | 3 | 3267 | 811.407431 | 165.034800 | 0.499479 | -1009.550772 | -33.608020 | 20.025 | 2074 | 471 | 722 |
| T01_EMA_9_21 | X2 | stress | 3 | 3267 | 1622.813452 | 814.066460 | 0.520417 | -2461.821424 | -25.461930 | 20.025 | 2063 | 821 | 383 |
| T02_ROC_20_60 | X1 | base | 3 | 8310 | 2064.059244 | 419.795570 | 0.927685 | -2388.868239 | 94.058890 | 8.166 | 4899 | 2355 | 1056 |
| T02_ROC_20_60 | X1 | stress | 3 | 8310 | 4074.816839 | 2044.077170 | 0.675472 | -6040.686507 | 77.532030 | 8.166 | 4742 | 3229 | 339 |
| T02_ROC_20_60 | X2 | base | 3 | 3671 | 911.770766 | 185.453860 | 0.864999 | -1015.919597 | 80.440030 | 19.450 | 2323 | 521 | 827 |
| T02_ROC_20_60 | X2 | stress | 3 | 3671 | 1823.533101 | 914.769990 | 0.872799 | -2652.272652 | 85.157640 | 19.450 | 2308 | 915 | 448 |
| T03_DONCHIAN_20 | X1 | base | 3 | 3258 | 809.210464 | 164.550860 | 1.000582 | -909.428932 | 63.331810 | 22.107 | 2116 | 433 | 709 |
| T03_DONCHIAN_20 | X1 | stress | 3 | 3258 | 1618.407446 | 811.845490 | 0.927057 | -2367.636259 | 61.689620 | 22.107 | 2108 | 738 | 412 |
| T03_DONCHIAN_20 | X2 | base | 3 | 3548 | 881.280272 | 179.227610 | 0.663714 | -1007.658268 | 52.185900 | 18.632 | 2256 | 511 | 781 |
| T03_DONCHIAN_20 | X2 | stress | 3 | 3548 | 1762.553657 | 884.142030 | 0.655455 | -2595.168581 | 50.871650 | 18.632 | 2236 | 885 | 427 |
| T04_ZSCORE_60 | X1 | base | 3 | 1024 | 254.310667 | 51.714500 | -0.397444 | -357.637981 | -51.215370 | 37.451 | 356 | 183 | 485 |
| T04_ZSCORE_60 | X1 | stress | 3 | 1024 | 508.621349 | 255.113800 | -0.397444 | -815.347963 | -51.215370 | 37.451 | 356 | 481 | 187 |
| T04_ZSCORE_60 | X2 | base | 3 | 1372 | 340.750503 | 69.280650 | 0.293067 | -503.095536 | -93.357450 | 22.436 | 851 | 171 | 350 |
| T04_ZSCORE_60 | X2 | stress | 3 | 1372 | 681.501034 | 341.843950 | 0.293067 | -1116.409367 | -93.357450 | 22.436 | 851 | 341 | 180 |

Diagnostics on consumed inner-screen blocks only; not G1/G2 evidence. Per-case observations overlap in time and are not independent samples. Terminal-censored cases are retained and are not completed round trips.

## T02 ROC/X1/Base signal and crossing detail

Counts below use one Base X1 run per consumed block. Crossings count adjacent valid READY rows whose nonzero sign changes; consensus counts transitions between agreeing positive and agreeing negative ROC signs, with neutral/disagreement states excluded from transition endpoints. Counts are path diagnostics, not independent samples.

| Block | Decision rows | READY | LONG_ENTRY | EXIT_LONG | SHORT_ENTRY | EXIT_SHORT | ROC20 sign flips | ROC60 sign flips | ROC consensus flips | Submitted orders | Filled orders | Native cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inner-screen-20240201_20240301 | 41757 | 41757 | 1741 | 1741 | 1517 | 1517 | 4436 | 2696 | 305 | 7470 | 7470 | 3258 |
| inner-screen-20240301_20240401 | 44637 | 44637 | 1752 | 1752 | 1714 | 1714 | 4758 | 2869 | 342 | 8350 | 8350 | 3466 |
| inner-screen-20240401_20240415 | 20158 | 20158 | 850 | 850 | 736 | 735 | 2179 | 1299 | 161 | 3171 | 3171 | 1586 |

Totals across blocks: {"EXIT_LONG": 4343, "EXIT_SHORT": 3966, "LONG_ENTRY": 4343, "NONE": 89933, "SHORT_ENTRY": 3967}; crossings: {"consensus": 808, "roc_20": 11373, "roc_60": 6864}.
