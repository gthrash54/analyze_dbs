# analyze_dbs

Measures and analyzes the Walker lab's intraoperative DBS evoked responses (ERNA). It starts from the post-processed
waveforms that [preprocess_dbs](https://github.com/gthrash54/preprocess_dbs) makes: time 0 at the artifact peak,
45 ms after it.

Work in progress. It follows the lab's MATLAB code for the ERNA parameters (latencies, amplitudes, frequency, RMS,
coastline burst index, z-scoring), with the lab's known bugs fixed.

## Install

```
git clone https://github.com/gthrash54/analyze_dbs
cd analyze_dbs
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
