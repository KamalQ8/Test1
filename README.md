# Edge Impulse Custom DSP Block (Stats + Band Energy)

Endpoints implemented per EI docs:
- `GET /` info
- `GET /parameters` returns `parameters.json`
- `POST /run` compute features for one sample
- `POST /batch` compute features for multiple samples

Features: mean, variance, min, max, RMS, zero-crossing rate, band energy in [low, high] Hz.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 4446
```

## Test with EI Blocks Runner
```bash
edge-impulse-blocks runner
# copy the public URL printed by the runner
```
Then Studio → Impulse design → Processing blocks → **Add custom block** → paste URL.

## Docker (optional)
```bash
docker build -t ei-custom-dsp .
docker run -p 4446:4446 ei-custom-dsp
# expose via ngrok if not using blocks runner
ngrok http 4446
```
