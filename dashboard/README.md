# Regulator Experiment Dashboard

A Streamlit-based dashboard for visualizing and analyzing regulator experiment episodes.

## Features

- **Price Trajectories**: Interactive plots showing individual firm prices and market price over time
- **Regulator Flags**: Visualization of monitoring results including parallel violations, structural break violations, and chat violations
- **Surplus Analysis**: Consumer surplus vs producer surplus analysis over time
- **Profit Analysis**: Individual firm profit trajectories
- **Episode Replay**: Framework for replaying episodes step-by-step
- **Data Export**: Download episode data as JSON

## Usage

1. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies** (if not already installed):
   ```bash
   pip install streamlit plotly
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

3. **Access the Dashboard**:
   Open your browser to `http://localhost:8501`

## Dashboard Components

### Episode Selection
- Select from available JSONL log files in the `logs/` directory
- View episode summary including agent types and environment parameters

### Visualizations

#### Price Trajectories Tab
- Individual firm price lines
- Market price line (dashed)
- Interactive hover information

#### Regulator Flags Tab
- Violation tracking over time (parallel, structural break, chat)
- Fines applied over time
- Subplot layout for easy comparison

#### Surplus Analysis Tab
- Consumer surplus (green, filled)
- Producer surplus (blue, filled)
- Economic welfare analysis

#### Profit Analysis Tab
- Individual firm profit trajectories
- Profit comparison over time

### Episode Replay
- Framework for step-by-step episode replay
- Export functionality for episode data

## Data Format

The dashboard expects JSONL log files with the following structure:

```json
{"type": "episode_header", "episode_id": "...", "n_firms": 2, ...}
{"type": "step", "step": 1, "prices": [...], "profits": [...], ...}
{"type": "episode_summary", "episode_id": "...", "total_reward": ...}
```

## Testing

Run the dashboard tests:

```bash
python -m pytest tests/test_dashboard.py -v
```

## Architecture

- **`app.py`**: Main Streamlit application
- **Plot Functions**: Modular functions for creating different visualizations
- **Data Loading**: JSONL file parsing and validation
- **Surplus Calculation**: Economic welfare analysis

## Dependencies

- `streamlit`: Web application framework
- `plotly`: Interactive plotting library
- `numpy`: Numerical computations
- `pathlib`: File system operations
