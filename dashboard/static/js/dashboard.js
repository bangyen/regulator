const COLORS = {
    primary: '#E63946',
    secondary: '#1D3557',
    tertiary: '#457B9D',
    quaternary: '#A8DADC',
    background: '#F8F9FA',
    text: '#1D1D1F',
    success: '#06D6A0'
};

const CHART_CONFIG = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
            labels: {
                font: { family: 'Space Grotesk', size: 12, weight: '500' },
                color: '#6E6E73',
                padding: 16,
                usePointStyle: true,
                pointStyle: 'circle'
            }
        }
    },
    scales: {
        x: {
            grid: { display: false, drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        },
        y: {
            grid: { color: '#E1E4E8', drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        }
    }
};

let charts = {};
let currentData = {};
let currentMetric = 'price';

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-container');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewName = item.getAttribute('data-view');
            
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            views.forEach(v => v.classList.add('hidden'));
            document.getElementById(`${viewName}-view`).classList.remove('hidden');
            
            // Update page title based on view
            const titles = {
                'overview': 'Dashboard Overview',
                'analytics': 'Analytics',
                'enforcement': 'Enforcement',
                'experiments': 'Experiments'
            };
            document.querySelector('.page-title').textContent = titles[viewName] || 'Dashboard';
            
            // Load view-specific data
            if (viewName === 'experiments') {
                loadExperiments();
            }
        });
    });
}

function initCharts() {
    // Main chart (price/profit toggle)
    const mainCtx = document.getElementById('main-chart')?.getContext('2d');
    if (mainCtx) {
        charts.main = new Chart(mainCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Market Price',
                    data: [],
                    borderColor: COLORS.primary,
                    backgroundColor: COLORS.primary + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0,
                    fill: true
                }]
            },
            options: CHART_CONFIG
        });
    }
    
    // Violations chart (bar)
    const violationsCtx = document.getElementById('violations-chart')?.getContext('2d');
    if (violationsCtx) {
        charts.violations = new Chart(violationsCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fines Issued',
                    data: [],
                    backgroundColor: COLORS.primary,
                    borderWidth: 0
                }]
            },
            options: {
                ...CHART_CONFIG,
                plugins: {
                    ...CHART_CONFIG.plugins,
                    legend: { display: false }
                }
            }
        });
    }
    
    // Cumulative fines chart
    const cumulativeCtx = document.getElementById('cumulative-chart')?.getContext('2d');
    if (cumulativeCtx) {
        charts.cumulative = new Chart(cumulativeCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative Fines',
                    data: [],
                    borderColor: COLORS.secondary,
                    backgroundColor: COLORS.secondary + '20',
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0,
                    fill: true
                }]
            },
            options: {
                ...CHART_CONFIG,
                aspectRatio: 2.5
            }
        });
    }
    
    // Price histogram
    const priceHistCtx = document.getElementById('price-histogram')?.getContext('2d');
    if (priceHistCtx) {
        charts.priceHist = new Chart(priceHistCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Frequency',
                    data: [],
                    backgroundColor: COLORS.tertiary,
                    borderWidth: 0
                }]
            },
            options: {
                ...CHART_CONFIG,
                plugins: {
                    ...CHART_CONFIG.plugins,
                    legend: { display: false }
                }
            }
        });
    }
    
    // Profit histogram
    const profitHistCtx = document.getElementById('profit-histogram')?.getContext('2d');
    if (profitHistCtx) {
        charts.profitHist = new Chart(profitHistCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Frequency',
                    data: [],
                    backgroundColor: COLORS.success,
                    borderWidth: 0
                }]
            },
            options: {
                ...CHART_CONFIG,
                plugins: {
                    ...CHART_CONFIG.plugins,
                    legend: { display: false }
                }
            }
        });
    }
}

function updateMetrics(metrics) {
    document.getElementById('avg-price').textContent = metrics.avg_price?.toFixed(2) || '—';
    document.getElementById('violations').textContent = metrics.total_violations || '—';
    document.getElementById('total-fines').textContent = metrics.total_fines?.toFixed(2) || '—';
    document.getElementById('risk-score').textContent = metrics.avg_risk_score?.toFixed(2) || '—';
}

function updateMainChart(timeSeries, metric = 'price') {
    if (!charts.main) return;
    
    const dataKey = metric === 'price' ? 'prices' : 'profits';
    const data = timeSeries[dataKey] || [];
    
    charts.main.data.labels = data.map(d => d.x);
    charts.main.data.datasets[0].data = data.map(d => d.y);
    charts.main.data.datasets[0].label = metric === 'price' ? 'Market Price' : 'Average Profit';
    charts.main.data.datasets[0].borderColor = metric === 'price' ? COLORS.primary : COLORS.success;
    charts.main.data.datasets[0].backgroundColor = (metric === 'price' ? COLORS.primary : COLORS.success) + '20';
    charts.main.update();
}

function updateViolationsChart(timeSeries) {
    if (!charts.violations) return;
    
    const data = timeSeries.fines || [];
    charts.violations.data.labels = data.map(d => d.x);
    charts.violations.data.datasets[0].data = data.map(d => d.y);
    charts.violations.update();
}

function updateCumulativeChart(timeSeries) {
    if (!charts.cumulative) return;
    
    const data = timeSeries.cumulative_fines || [];
    charts.cumulative.data.labels = data.map(d => d.x);
    charts.cumulative.data.datasets[0].data = data.map(d => d.y);
    charts.cumulative.update();
}

function createHistogram(data, bins = 10) {
    if (!data || data.length === 0) return { labels: [], values: [] };
    
    const values = data.map(d => d.y);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binSize = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    values.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binSize), bins - 1);
        histogram[binIndex]++;
    });
    
    const labels = [];
    for (let i = 0; i < bins; i++) {
        const start = (min + i * binSize).toFixed(1);
        const end = (min + (i + 1) * binSize).toFixed(1);
        labels.push(`${start}-${end}`);
    }
    
    return { labels, values: histogram };
}

function updateHistograms(timeSeries) {
    // Price histogram
    if (charts.priceHist && timeSeries.prices) {
        const hist = createHistogram(timeSeries.prices, 8);
        charts.priceHist.data.labels = hist.labels;
        charts.priceHist.data.datasets[0].data = hist.values;
        charts.priceHist.update();
    }
    
    // Profit histogram
    if (charts.profitHist && timeSeries.profits) {
        const hist = createHistogram(timeSeries.profits, 8);
        charts.profitHist.data.labels = hist.labels;
        charts.profitHist.data.datasets[0].data = hist.values;
        charts.profitHist.update();
    }
}

function updateTable(timeSeries) {
    const tbody = document.getElementById('activity-table');
    if (!tbody) return;
    
    const recentData = (timeSeries.prices || []).slice(-15).reverse();
    
    if (recentData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading">No data available</td></tr>';
        return;
    }
    
    tbody.innerHTML = recentData.map(pricePoint => {
        const step = pricePoint.x;
        const price = pricePoint.y.toFixed(2);
        const profitPoint = timeSeries.profits.find(p => p.x === step);
        const profit = profitPoint ? profitPoint.y.toFixed(2) : '—';
        const finePoint = timeSeries.fines.find(f => f.x === step);
        const fine = finePoint ? finePoint.y.toFixed(2) : '0.00';
        const isViolation = finePoint && finePoint.y > 0;
        
        return `
            <tr>
                <td><strong>${step}</strong></td>
                <td>${price}</td>
                <td>${profit}</td>
                <td><span class="violation-badge ${isViolation ? 'yes' : 'no'}">${isViolation ? 'Yes' : 'No'}</span></td>
                <td>${fine}</td>
            </tr>
        `;
    }).join('');
}

function loadExperiments() {
    fetch('/api/experiments')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('experiments-list');
            if (!container) return;
            
            if (data.length === 0) {
                container.innerHTML = '<p class="loading">No experiments found</p>';
                return;
            }
            
            container.innerHTML = data.map(exp => {
                const date = new Date(exp.modified);
                const dateStr = date.toLocaleString();
                return `
                    <div class="experiment-item">
                        <div class="experiment-name">${exp.name}</div>
                        <div class="experiment-date">${dateStr}</div>
                    </div>
                `;
            }).join('');
        })
        .catch(err => {
            console.error('Failed to load experiments:', err);
            const container = document.getElementById('experiments-list');
            if (container) {
                container.innerHTML = '<p class="loading">Error loading experiments</p>';
            }
        });
}

async function fetchData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) {
            throw new Error('No data available');
        }
        const data = await response.json();
        
        currentData = data;
        updateMetrics(data.metrics);
        updateMainChart(data.time_series, currentMetric);
        updateViolationsChart(data.time_series);
        updateCumulativeChart(data.time_series);
        updateHistograms(data.time_series);
        updateTable(data.time_series);
        
        // Update sidebar status
        document.getElementById('sidebar-status').textContent = 'Live';
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('sidebar-status').textContent = 'Error';
    }
}

function initToggleButtons() {
    document.querySelectorAll('[data-metric]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-metric]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMetric = btn.getAttribute('data-metric');
            
            if (currentData.time_series) {
                updateMainChart(currentData.time_series, currentMetric);
            }
        });
    });
}

function initExperimentRunner() {
    const runBtn = document.getElementById('run-btn');
    if (!runBtn) return;
    
    runBtn.addEventListener('click', async () => {
        try {
            runBtn.disabled = true;
            runBtn.classList.add('running');
            runBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="2"/>
                </svg>
                Running...
            `;
            
            document.getElementById('sidebar-status').textContent = 'Running';
            
            const response = await fetch('/api/experiment/run', { method: 'POST' });
            if (!response.ok) {
                throw new Error('Failed to start experiment');
            }
            
            // Poll for status
            const statusInterval = setInterval(async () => {
                const statusRes = await fetch('/api/experiment/status');
                const status = await statusRes.json();
                
                if (status.status === 'completed') {
                    clearInterval(statusInterval);
                    runBtn.disabled = false;
                    runBtn.classList.remove('running');
                    runBtn.innerHTML = `
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <polygon points="4 2 14 8 4 14 4 2" fill="currentColor"/>
                        </svg>
                        Run Experiment
                    `;
                    document.getElementById('sidebar-status').textContent = 'Live';
                    fetchData(); // Refresh data
                } else if (status.status === 'error') {
                    clearInterval(statusInterval);
                    runBtn.disabled = false;
                    runBtn.classList.remove('running');
                    runBtn.innerHTML = `
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <polygon points="4 2 14 8 4 14 4 2" fill="currentColor"/>
                        </svg>
                        Run Experiment
                    `;
                    document.getElementById('sidebar-status').textContent = 'Error';
                    const errorMsg = status.error_message || 'Unknown error';
                    alert('Experiment failed: ' + errorMsg);
                }
            }, 2000);
            
        } catch (error) {
            console.error('Error running experiment:', error);
            runBtn.disabled = false;
            runBtn.classList.remove('running');
            runBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <polygon points="4 2 14 8 4 14 4 2" fill="currentColor"/>
                </svg>
                Run Experiment
            `;
            document.getElementById('sidebar-status').textContent = 'Error';
            alert('Failed to start experiment');
        }
    });
}

function initExportButton() {
    const exportBtn = document.getElementById('export-btn');
    if (!exportBtn) return;
    
    exportBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/data');
            const data = await response.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `regulator-data-${new Date().toISOString()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error exporting data:', error);
            alert('Failed to export data');
        }
    });
}

function initRefreshButton() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (!refreshBtn) return;
    
    refreshBtn.addEventListener('click', () => {
        fetchData();
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCharts();
    initToggleButtons();
    initExperimentRunner();
    initExportButton();
    initRefreshButton();
    
    // Load initial data
    fetchData();
    
    // Auto-refresh every 30 seconds
    setInterval(fetchData, 30000);
});
