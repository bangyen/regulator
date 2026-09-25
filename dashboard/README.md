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
pip install -e ".[dashboard]"

# Run dashboard
python dashboard/main.py

# Or use Just
just dashboard
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
- **Run** - Pick up to three firms, the regulator, steps and a seed (blank =
  random), then click Run Experiment
- **Refresh** - Data reloads every 30 seconds and after each run
- **Switch Views** - Toggle the main chart between Price and Profit
- **Export Data** - Click Export (Enforcement view) to download JSON

Logs are read from `logs/` at the repo root; set `REGULATOR_LOG_DIR` to use
another directory. `DASHBOARD_HOST`, `DASHBOARD_PORT` and `FLASK_DEBUG`
control the server.

## Features

- **Real-time Monitoring** - Auto-refreshes every 30 seconds
- **Key Metrics** - Price, violations, fines, risk scores
- **Interactive Charts** - Switchable price/profit views
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

