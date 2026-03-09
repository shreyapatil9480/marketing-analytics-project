# Marketing Analytics Project

What drives logistics delivery delays?

**Stakeholder:** VP Supply Chain

## Key Insights

- Route miles above 900 increase delay probability by 18%.
- Hub congestion spikes delays on Fridays regardless of carrier.
- Weather delay flags alone explain 12% of late deliveries.

## Dataset

Primary file: `data/raw/shipment_delays.csv`  
Target variable: `delayed`

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook notebooks/eda.ipynb
```

## Next Steps

**Done.** A/B channel lift analysis is implemented — see ### Implemented below.

---
*Analytics portfolio project — 2025-07*

<!-- build 4 -->

### Implemented

```bash
pip install -r requirements.txt
python scripts/ab_lift.py
```