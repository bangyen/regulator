# Regulator Dashboard

Professional monitoring interface for real-time cartel detection and market analysis.

## Design

Minimalist Bauhaus-inspired design with:
- Space Grotesk typography
- Flat color palette (red, navy, cyan, blue, orange, green)
- Custom SVG icons
- Canvas-based charts
- Sharp geometric layouts

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies (if not already done)
pip install -e .

# Run dashboard
python dashboard/main.py

# Or use Make
make dashboard
```

Visit `http://localhost:5000` in your browser.

## Usage

### 1. Run an Experiment
First, generate some data to visualize:

```bash
python scripts/run_experiment.py --steps 100 --firms "random,tit_for_tat"
```

### 2. Start Dashboard
The dashboard will automatically load the most recent experiment:

```bash
python dashboard/main.py
```

### 3. Interact
- **Refresh** - Click refresh button or wait for auto-update (5s)
- **Switch Views** - Toggle between Price and Risk charts
- **Export Data** - Click Export button to download JSON

## Features

- **Real-time Monitoring** - Auto-refreshes every 5 seconds
- **Key Metrics** - Price, violations, fines, risk scores
- **Interactive Charts** - Switchable price/risk views
- **Activity Table** - Recent step-by-step data
- **Data Export** - Download JSON snapshots

## API Endpoints

- `GET /` - Dashboard interface
- `GET /api/data` - Current metrics and time series
- `GET /api/experiments` - List available experiments

## Development

The dashboard reads experiment logs from `/logs/*.jsonl` and displays the most recent data.

### File Structure
```
dashboard/
├── main.py              # Flask application
├── templates/
│   └── dashboard.html   # Main template
├── static/
│   ├── css/
│   │   └── style.css    # Design system
│   └── js/
│       └── dashboard.js # Chart rendering + API client
├── README.md            # This file
├── DESIGN.md            # Design system documentation
└── FEATURES.md          # Feature details
```

### Testing
```bash
# Run dashboard tests
pytest tests/unit/test_dashboard.py -v
pytest tests/integration/test_dashboard_integration.py -v
```

