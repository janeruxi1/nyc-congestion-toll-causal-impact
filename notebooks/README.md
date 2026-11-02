# Notebooks: reading guide

Read them in order. Each notebook opens with its question and a short answer, and every figure
has a short explanation next to it.

| # | Notebook | The question in plain words | Main figure |
|---|---|---|---|
| 01 | [Data audit](01_data_audit.ipynb) | Can the trip records be trusted, and which zones are in the toll zone? | Rows kept; zones by share paying the fee |
| 02 | [Trends](02_trends.ipynb) | What does the raw data show before any model? | CBD share before and after; trips by hour |
| 03 | [Difference-in-differences](03_difference_in_differences.ipynb) | Did the toll reduce trips compared with the rest of Manhattan? | Event study; fake toll dates; other outcomes |
| 04 | [Forecast counterfactual](04_forecast_counterfactual.ipynb) | Did 2025 fall short of a no-toll forecast? | Forecast band vs actual trips |
| 05 | [Synthetic control](05_synthetic_control.ipynb) | Is there a gap against a "CBD without the toll"? | Gap vs synthetic twin; placebo zones |
| 06 | [Results summary](06_results_summary.ipynb) | What do the three methods say together? | All estimates side by side |
| 07 | [Robust inference](07_robust_inference.ipynb) | How much can the intervals be trusted? | HonestDiD sensitivity |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source); see the main [README](../README.md) for the data and the build command.
