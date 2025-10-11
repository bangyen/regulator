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
        this.data = data;
        this.draw();
    }

    draw() {
        if (!this.data || this.data.length === 0) {
            this.drawEmpty();
            return;
        }

        this.ctx.clearRect(0, 0, this.width, this.height);
        
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
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i++) {
            const y = this.padding + (chartHeight / 4) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(this.padding, y);
            this.ctx.lineTo(this.padding + chartWidth, y);
            this.ctx.stroke();
        }
    }

    drawLine(chartWidth, chartHeight, minX, maxX, minY, maxY, yRange) {
        const xScale = chartWidth / (maxX - minX || 1);
        const yScale = chartHeight / yRange;

        this.ctx.beginPath();
        this.data.forEach((point, i) => {
            const x = this.padding + (point.x - minX) * xScale;
            const y = this.padding + chartHeight - (point.y - minY) * yScale;
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });

        this.ctx.strokeStyle = this.colors.line;
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        this.ctx.lineTo(this.padding + chartWidth, this.padding + chartHeight);
        this.ctx.lineTo(this.padding, this.padding + chartHeight);
        this.ctx.closePath();
        this.ctx.fillStyle = this.colors.fill + '40';
        this.ctx.fill();
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
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.padding, this.padding);
        this.ctx.lineTo(this.padding, this.padding + chartHeight);
        this.ctx.lineTo(this.padding + chartWidth, this.padding + chartHeight);
        this.ctx.stroke();

        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = '600 11px Space Grotesk';
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'middle';

        for (let i = 0; i <= 4; i++) {
            const value = maxY - (maxY - minY) * (i / 4);
            const y = this.padding + (chartHeight / 4) * i;
            this.ctx.fillText(value.toFixed(1), this.padding - 10, y);
        }
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

const priceChart = new DashboardChart('price-chart', 'line');
const riskChart = new DashboardChart('price-chart', 'line');
const violationsChart = new DashboardChart('violations-chart', 'bar');
const finesChart = new DashboardChart('fines-chart', 'bar');

let currentChart = 'price';

function updateMetrics(metrics) {
    document.getElementById('avg-price').textContent = metrics.avg_price.toFixed(2);
    document.getElementById('violations').textContent = metrics.total_violations;
    document.getElementById('total-fines').textContent = metrics.total_fines.toFixed(2);
    document.getElementById('risk-score').textContent = metrics.current_risk.toFixed(2);
}

function updateCharts(timeSeries) {
    priceChart.setData(timeSeries.prices);
    riskChart.setData(timeSeries.risk_scores);
    violationsChart.setData(timeSeries.violations);
    finesChart.setData(timeSeries.fines);
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
        const riskPoint = timeSeries.risk_scores.find(r => r.x === step);
        const risk = riskPoint ? riskPoint.y.toFixed(2) : '—';
        const violationPoint = timeSeries.violations.find(v => v.x === step);
        const isViolation = violationPoint && violationPoint.y === 1;
        const finePoint = timeSeries.fines.find(f => f.x === step);
        const fine = finePoint ? finePoint.y.toFixed(2) : '0.00';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${step}</strong></td>
            <td>${price}</td>
            <td>${risk}</td>
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
        document.getElementById('status-text').textContent = 'Live';
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('status-text').textContent = 'Error';
    }
}

document.getElementById('refresh-btn').addEventListener('click', fetchData);

document.querySelectorAll('[data-chart]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('[data-chart]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentChart = btn.dataset.chart;
        
        if (currentChart === 'price') {
            priceChart.resizeCanvas();
        } else if (currentChart === 'risk') {
            riskChart.resizeCanvas();
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
setInterval(fetchData, 5000);

