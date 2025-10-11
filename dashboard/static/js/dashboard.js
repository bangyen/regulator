class DashboardChart {
    constructor(canvasId, type = 'line') {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.type = type;
        this.data = [];
        this.padding = 40;
        this.colors = {
            line: '#1D3557',
            fill: '#A8DADC',
            grid: '#E5E5E5',
            text: '#A3A3A3'
        };
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.width = rect.width;
        this.height = rect.height;
        this.draw();
    }

    setData(data) {
        // Filter out invalid/empty data points
        this.data = data.filter(point => {
            return point && typeof point.x === 'number' && typeof point.y === 'number' 
                   && !isNaN(point.x) && !isNaN(point.y);
        });
        this.draw();
    }

    draw() {
        if (!this.data || this.data.length === 0) {
            this.drawEmpty();
            return;
        }

        // Clear entire canvas properly
        this.ctx.save();
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.restore();
        
        const chartWidth = this.width - 2 * this.padding;
        const chartHeight = this.height - 2 * this.padding;
        
        const xValues = this.data.map(d => d.x);
        const yValues = this.data.map(d => d.y);
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        const minY = Math.min(...yValues, 0);
        const maxY = Math.max(...yValues);
        const yRange = maxY - minY || 1;

        this.drawGrid(chartWidth, chartHeight, minY, maxY);

        if (this.type === 'bar') {
            this.drawBars(chartWidth, chartHeight, minX, maxX, minY, maxY, yRange);
        } else {
            this.drawLine(chartWidth, chartHeight, minX, maxX, minY, maxY, yRange);
        }

        this.drawAxes(chartWidth, chartHeight, minY, maxY);
    }

    drawGrid(chartWidth, chartHeight, minY, maxY) {
        this.ctx.save();
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i++) {
            const y = this.padding + (chartHeight / 4) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(this.padding, y);
            this.ctx.lineTo(this.padding + chartWidth, y);
            this.ctx.stroke();
        }
        this.ctx.beginPath(); // Clear path
        this.ctx.restore();
    }

    drawLine(chartWidth, chartHeight, minX, maxX, minY, maxY, yRange) {
        const xScale = chartWidth / (maxX - minX || 1);
        const yScale = chartHeight / yRange;

        // Draw each line segment individually
        this.ctx.save();
        this.ctx.strokeStyle = this.colors.line;
        this.ctx.lineWidth = 2;
        
        for (let i = 0; i < this.data.length - 1; i++) {
            const point = this.data[i];
            const nextPoint = this.data[i + 1];
            
            const x1 = this.padding + (point.x - minX) * xScale;
            const y1 = this.padding + chartHeight - (point.y - minY) * yScale;
            const x2 = this.padding + (nextPoint.x - minX) * xScale;
            const y2 = this.padding + chartHeight - (nextPoint.y - minY) * yScale;
            
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();
        }
        
        this.ctx.restore();
    }

    drawBars(chartWidth, chartHeight, minX, maxX, minY, maxY, yRange) {
        const barWidth = Math.max(2, chartWidth / this.data.length * 0.8);
        const xScale = chartWidth / (maxX - minX || 1);
        const yScale = chartHeight / yRange;

        this.data.forEach(point => {
            const x = this.padding + (point.x - minX) * xScale - barWidth / 2;
            const barHeight = (point.y - minY) * yScale;
            const y = this.padding + chartHeight - barHeight;

            this.ctx.fillStyle = point.y > 0 ? '#E63946' : this.colors.fill;
            this.ctx.fillRect(x, y, barWidth, barHeight);
        });
    }

    drawAxes(chartWidth, chartHeight, minY, maxY) {
        this.ctx.save();
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.padding, this.padding);
        this.ctx.lineTo(this.padding, this.padding + chartHeight);
        this.ctx.lineTo(this.padding + chartWidth, this.padding + chartHeight);
        this.ctx.stroke();
        this.ctx.beginPath(); // Clear path

        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = '600 11px Space Grotesk';
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'middle';

        for (let i = 0; i <= 4; i++) {
            const value = maxY - (maxY - minY) * (i / 4);
            const y = this.padding + (chartHeight / 4) * i;
            this.ctx.fillText(value.toFixed(1), this.padding - 10, y);
        }
        this.ctx.restore();
    }

    drawEmpty() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = '14px Space Grotesk';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('No data available', this.width / 2, this.height / 2);
    }
}

const mainChart = new DashboardChart('price-chart', 'line');
const violationsChart = new DashboardChart('violations-chart', 'bar');
const cumulativeChart = new DashboardChart('cumulative-chart', 'line');

let currentChart = 'price';
let currentData = { prices: [], profits: [] };

function updateMetrics(metrics) {
    document.getElementById('avg-price').textContent = metrics.avg_price.toFixed(2);
    document.getElementById('violations').textContent = metrics.total_violations;
    document.getElementById('total-fines').textContent = metrics.total_fines.toFixed(2);
    document.getElementById('risk-score').textContent = metrics.avg_risk_score.toFixed(2);
}

function updateCharts(timeSeries) {
    currentData = timeSeries;
    if (currentChart === 'price') {
        mainChart.setData(timeSeries.prices);
    } else {
        mainChart.setData(timeSeries.profits);
    }
    violationsChart.setData(timeSeries.violations);
    cumulativeChart.setData(timeSeries.cumulative_fines);
}

function updateTable(timeSeries) {
    const tbody = document.getElementById('activity-table');
    tbody.innerHTML = '';

    const recentData = timeSeries.prices.slice(-10).reverse();
    
    if (recentData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No data available</td></tr>';
        return;
    }

    recentData.forEach((pricePoint, idx) => {
        const step = pricePoint.x;
        const price = pricePoint.y.toFixed(2);
        const profitPoint = timeSeries.profits.find(p => p.x === step);
        const profit = profitPoint ? profitPoint.y.toFixed(2) : '—';
        const finePoint = timeSeries.fines.find(f => f.x === step);
        const fine = finePoint ? finePoint.y.toFixed(2) : '0.00';
        const isViolation = finePoint && finePoint.y > 0;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${step}</strong></td>
            <td>${price}</td>
            <td>${profit}</td>
            <td><span class="violation-badge ${isViolation ? 'yes' : 'no'}">${isViolation ? 'Yes' : 'No'}</span></td>
            <td>${fine}</td>
        `;
        tbody.appendChild(row);
    });
}

async function fetchData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) {
            throw new Error('No data available');
        }
        const data = await response.json();
        updateMetrics(data.metrics);
        updateCharts(data.time_series);
        updateTable(data.time_series);
        document.getElementById('status-text').textContent = 'Ready';
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('status-text').textContent = 'Error';
    }
}

document.getElementById('run-btn').addEventListener('click', async () => {
    const btn = document.getElementById('run-btn');
    try {
        btn.disabled = true;
        btn.classList.add('running');
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
            </svg>
            Running...
        `;
        
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
                btn.disabled = false;
                btn.classList.remove('running');
                btn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    Run Experiment
                `;
                fetchData(); // Refresh data
            } else if (status.status === 'error') {
                clearInterval(statusInterval);
                btn.disabled = false;
                btn.classList.remove('running');
                btn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    Run Experiment
                `;
                const errorMsg = status.error_message || 'Unknown error';
                alert('Experiment failed: ' + errorMsg);
            }
        }, 2000);
        
    } catch (error) {
        console.error('Error running experiment:', error);
        btn.disabled = false;
        btn.classList.remove('running');
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            Run Experiment
        `;
        alert('Failed to start experiment');
    }
});

document.querySelectorAll('[data-chart]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('[data-chart]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentChart = btn.dataset.chart;
        
        if (currentChart === 'price') {
            mainChart.setData(currentData.prices);
        } else if (currentChart === 'profit') {
            mainChart.setData(currentData.profits);
        }
    });
});

document.getElementById('export-btn').addEventListener('click', async () => {
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
    }
});

fetchData();

