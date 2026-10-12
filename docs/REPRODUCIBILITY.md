# Reproduction Requirements

Dataset: **In preparation**. This repository publishes source code and protocol
definitions. It does not distribute the data needed to regenerate the paper's
numerical results. No Hugging Face upload or download endpoint is configured.

## Code Checks

```sh
python -m unittest discover -s tools -p 'test_code_release.py'
```

The tests check pure scoring functions, parser availability, protocol constants,
and the unavailable-data behavior. They use synthetic values only. They do not
verify the paper's measured scores or run a model or game simulation.

## Analysis Inputs

| Analysis | Required local inputs, not included |
| --- | --- |
| Main and detailed results | Matching score summaries, scored records, and benchmark calibration artifacts |
| Aligned reasoning | Aligned questions, responses, scoring records, and process evidence |
| Usability | Full-game and local paired-execution records |
| External comparisons | Frozen external comparison records and matched aggregate scores |
| Scorer verification | Questions, model responses, reference scored records, and calibration files |

The native directory and field contract is described in [PROTOCOL.md](PROTOCOL.md).
`tools/reproduce.py` retains the analysis stage implementations for use with
their corresponding inputs. It checks for the required release index before
creating output directories. `tools/unpack.py` can verify a separately supplied
local archive index; no archive index is supplied here. `tools/download_data.py`
has no network implementation in this release.

For independent experiments, the collection, construction, inference, and
scoring programs accept their own inputs. Use a separate run directory and output
tag. Engine collection and model inference are distinct from CPU analysis and
require their own dependencies, data, and compute. Full paper reproduction will
need a separately announced, versioned data release.
