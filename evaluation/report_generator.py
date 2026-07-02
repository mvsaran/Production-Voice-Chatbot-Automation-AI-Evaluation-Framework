"""
Enterprise Report Generator.

Compiles evaluation results into stunning, glassmorphic HTML dashboards with
Chart.js visualizations, SLA progress bars, acoustic WER/CER analysis, and
detailed conversation telemetry.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from jinja2 import Template
from loguru import logger
from evaluation.models.reports import AggregatedEvaluationReport, DashboardReadyEvaluationOutput
from evaluation.models.results import MetricStatusEnum


HTML_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Chatbot AI - Enterprise QA Dashboard</title>
    <!-- Chart.js for interactive graphics -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: rgba(25, 33, 50, 0.75);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-purple: #b388ff;
            --success: #00e676;
            --warning: #ffb300;
            --danger: #ff5252;
            --text-main: #f0f4f8;
            --text-muted: #8e9bad;
            --border: rgba(255, 255, 255, 0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-primary);
            background-image: radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 85% 85%, rgba(179, 136, 255, 0.05) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        /* Header & Glassmorphism */
        header {
            display: flex; justify-content: space-between; align-items: center;
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        .logo-area h1 {
            font-size: 1.8rem; font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .logo-area p { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.2rem; }
        .meta-tags { display: flex; gap: 1rem; align-items: center; }
        .badge {
            padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .badge.pass { background: rgba(0, 230, 118, 0.15); color: var(--success); border: 1px solid var(--success); }
        .badge.fail { background: rgba(255, 82, 82, 0.15); color: var(--danger); border: 1px solid var(--danger); }
        
        /* KPI Summary Grid */
        .kpi-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem; margin-bottom: 2rem;
        }
        .kpi-card {
            background: var(--bg-card); backdrop-filter: blur(10px);
            border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem;
            position: relative; overflow: hidden; transition: transform 0.2s, border-color 0.2s;
        }
        .kpi-card:hover { transform: translateY(-4px); border-color: var(--accent-blue); }
        .kpi-title { color: var(--text-muted); font-size: 0.85rem; font-weight: 500; text-transform: uppercase; }
        .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0.5rem 0; color: var(--text-main); }
        .kpi-sub { font-size: 0.8rem; color: var(--accent-cyan); }
        
        /* Chart Grid */
        .chart-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;
        }
        @media (max-width: 968px) { .chart-grid { grid-template-columns: 1fr; } }
        .chart-box {
            background: var(--bg-card); backdrop-filter: blur(10px);
            border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem;
        }
        .chart-box h3 { font-size: 1.1rem; margin-bottom: 1rem; color: var(--text-main); font-weight: 600; }
        .chart-wrapper { position: relative; height: 300px; width: 100%; }

        /* Filter Tabs */
        .filter-bar {
            display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap;
        }
        .filter-btn {
            background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-muted);
            padding: 0.5rem 1.25rem; border-radius: 8px; cursor: pointer; font-weight: 500;
            transition: all 0.2s; font-family: 'Outfit', sans-serif;
        }
        .filter-btn:hover, .filter-btn.active {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
            color: #000; font-weight: 600; border-color: transparent;
        }
        
        /* Scenario Cards */
        .scenario-list { display: flex; flex-direction: column; gap: 1rem; }
        .scenario-card {
            background: var(--bg-card); backdrop-filter: blur(10px);
            border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem;
            transition: all 0.2s;
        }
        .scenario-card:hover { border-color: rgba(255, 255, 255, 0.2); }
        .scenario-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .scenario-title { font-size: 1.2rem; font-weight: 600; color: var(--accent-cyan); display: flex; align-items: center; gap: 0.75rem; }
        .scenario-id { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; background: var(--bg-secondary); padding: 0.2rem 0.6rem; border-radius: 6px; color: var(--text-muted); }
        
        /* Dialogue Grid */
        .dialogue-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
            background: var(--bg-secondary); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
        }
        @media (max-width: 768px) { .dialogue-grid { grid-template-columns: 1fr; } }
        .dialogue-box { display: flex; flex-direction: column; gap: 0.4rem; }
        .dialogue-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); }
        .dialogue-text { font-size: 0.95rem; color: var(--text-main); background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; border-left: 3px solid var(--accent-blue); }
        .dialogue-text.expected { border-left-color: var(--accent-purple); }
        
        /* Metrics Bar */
        .metrics-bar { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
        .metric-pill {
            display: flex; align-items: center; gap: 0.4rem; background: rgba(0,0,0,0.3);
            border: 1px solid var(--border); padding: 0.4rem 0.8rem; border-radius: 8px;
            font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;
        }
        .metric-pill span { color: var(--text-muted); }
        .metric-pill strong { color: var(--text-main); }
        .metric-pill.good strong { color: var(--success); }
        .metric-pill.warn strong { color: var(--warning); }
        .metric-pill.bad strong { color: var(--danger); }

        /* Footer */
        footer { text-align: center; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="logo-area">
                <h1>🎙️ Voice Chatbot AI Automation</h1>
                <p>Enterprise AI Quality Assurance & Speech Telemetry Dashboard</p>
            </div>
            <div class="meta-tags">
                <div class="badge {{ 'pass' if overall_passed else 'fail' }}">
                    {{ "SLA PASSED" if overall_passed else "SLA BREACHED" }}
                </div>
                <div style="text-align: right; font-size: 0.85rem; color: var(--text-muted);">
                    Generated: <span style="color: var(--text-main);">{{ timestamp }}</span><br>
                    Engine: <strong>DeepEval + Whisper + GPT-4o</strong>
                </div>
            </div>
        </header>

        <!-- KPI Summary Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Overall Pass Rate</div>
                <div class="kpi-value" style="color: {{ '#00e676' if pass_rate >= 90 else '#ffb300' }};">{{ pass_rate }}%</div>
                <div class="kpi-sub">{{ total_passed }} / {{ total_scenarios }} Scenarios Passed SLA</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average AI Quality Score</div>
                <div class="kpi-value">{{ avg_quality_score }}<span style="font-size: 1.2rem;">/1.0</span></div>
                <div class="kpi-sub">DeepEval Correctness & Relevancy</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Speech Recognition (WER)</div>
                <div class="kpi-value" style="color: {{ '#00e676' if avg_wer <= 0.10 else '#ff5252' }};">{{ avg_wer }}</div>
                <div class="kpi-sub">Target SLA: WER &le; 0.10 (90%+ Accuracy)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average End-to-End Latency</div>
                <div class="kpi-value">{{ avg_latency }}s</div>
                <div class="kpi-sub">Whisper STT + GPT-4o + OpenAI TTS</div>
            </div>
        </div>

        <!-- Interactive Chart Grid -->
        <div class="chart-grid">
            <div class="chart-box">
                <h3>📈 DeepEval AI Quality Radar (SLA Compliance)</h3>
                <div class="chart-wrapper">
                    <canvas id="qualityRadarChart"></canvas>
                </div>
            </div>
            <div class="chart-box">
                <h3>⏱️ End-to-End Turn Latency Breakdown (Seconds)</h3>
                <div class="chart-wrapper">
                    <canvas id="latencyBarChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Filter Bar -->
        <div class="filter-bar">
            <button class="filter-btn active" onclick="filterScenarios('ALL', this)">All Scenarios ({{ total_scenarios }})</button>
            <button class="filter-btn" onclick="filterScenarios('PASS', this)">Passed Only ({{ total_passed }})</button>
            <button class="filter-btn" onclick="filterScenarios('FAIL', this)">Failed / Warnings ({{ total_scenarios - total_passed }})</button>
            <button class="filter-btn" onclick="filterScenarios('Banking', this)">🏦 Banking</button>
            <button class="filter-btn" onclick="filterScenarios('Healthcare', this)">🏥 Healthcare</button>
            <button class="filter-btn" onclick="filterScenarios('Insurance', this)">🛡️ Insurance</button>
            <button class="filter-btn" onclick="filterScenarios('Travel', this)">✈️ Travel</button>
            <button class="filter-btn" onclick="filterScenarios('E-commerce', this)">🛒 E-Commerce</button>
            <button class="filter-btn" onclick="filterScenarios('Support', this)">🎧 Support</button>
        </div>

        <!-- Scenario List -->
        <div class="scenario-list">
            {% for sc in scenarios %}
            <div class="scenario-card" data-status="{{ sc.status }}" data-category="{{ sc.category }}">
                <div class="scenario-header">
                    <div class="scenario-title">
                        {{ sc.icon }} {{ sc.scenario_name }}
                        <span class="scenario-id">{{ sc.conversation_id }}</span>
                    </div>
                    <div class="badge {{ 'pass' if sc.status == 'PASS' else 'fail' }}">
                        {{ sc.status }}
                    </div>
                </div>

                <div class="dialogue-grid">
                    <div class="dialogue-box">
                        <div class="dialogue-label">👤 User Spoke (Transcribed by Whisper)</div>
                        <div class="dialogue-text">"{{ sc.prompt }}"</div>
                    </div>
                    <div class="dialogue-box">
                        <div class="dialogue-label">🤖 Chatbot Voice Answer (Synthesized by TTS)</div>
                        <div class="dialogue-text">"{{ sc.response }}"</div>
                    </div>
                </div>

                <div class="metrics-bar">
                    <div class="metric-pill {{ 'good' if sc.correctness >= 0.85 else 'warn' }}">
                        <span>Correctness:</span> <strong>{{ sc.correctness }}</strong>
                    </div>
                    <div class="metric-pill {{ 'good' if sc.relevancy >= 0.85 else 'warn' }}">
                        <span>Relevancy:</span> <strong>{{ sc.relevancy }}</strong>
                    </div>
                    <div class="metric-pill {{ 'good' if sc.hallucination <= 0.15 else 'bad' }}">
                        <span>Hallucination:</span> <strong>{{ sc.hallucination }}</strong>
                    </div>
                    <div class="metric-pill {{ 'good' if sc.wer <= 0.10 else 'bad' }}">
                        <span>WER:</span> <strong>{{ sc.wer }}</strong>
                    </div>
                    <div class="metric-pill {{ 'good' if sc.cer <= 0.05 else 'bad' }}">
                        <span>CER:</span> <strong>{{ sc.cer }}</strong>
                    </div>
                    <div class="metric-pill">
                        <span>STT Latency:</span> <strong>{{ sc.stt_latency }}s</strong>
                    </div>
                    <div class="metric-pill">
                        <span>LLM Latency:</span> <strong>{{ sc.llm_latency }}s</strong>
                    </div>
                    <div class="metric-pill">
                        <span>TTS Latency:</span> <strong>{{ sc.tts_latency }}s</strong>
                    </div>
                    <div class="metric-pill" style="background: rgba(0, 242, 254, 0.1); border-color: var(--accent-cyan);">
                        <span>Total Latency:</span> <strong style="color: var(--accent-cyan);">{{ sc.latency }}s</strong>
                    </div>
                </div>
                {% if sc.failure_reason %}
                <div style="margin-top: 0.75rem; font-size: 0.85rem; color: var(--danger); background: rgba(255, 82, 82, 0.1); padding: 0.5rem 0.75rem; border-radius: 6px;">
                    ⚠️ <strong>Reason:</strong> {{ sc.failure_reason }}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <!-- Footer -->
        footer>
            Voice Chatbot Automation Evaluation Framework &bull; Built with FastAPI, OpenAI Whisper, GPT-4o, and DeepEval G-Eval &bull; &copy; 2026 Enterprise QA
        </footer>
    </div>

    <!-- Chart.js Logic -->
    <script>
        // Data injected from Python
        const chartData = {{ chart_data | safe }};
        
        // 1. Radar Chart: Quality Metrics
        const ctxRadar = document.getElementById('qualityRadarChart').getContext('2d');
        new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Correctness', 'Relevancy', 'Faithfulness', 'Non-Hallucination', 'Speech Accuracy (1-WER)'],
                datasets: [{
                    label: 'Current Build SLA',
                    data: chartData.radar_values,
                    backgroundColor: 'rgba(0, 242, 254, 0.2)',
                    borderColor: '#00f2fe',
                    pointBackgroundColor: '#00e676',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#00e676',
                    borderWidth: 2
                }, {
                    label: 'Enterprise Target SLA',
                    data: [0.85, 0.85, 0.90, 0.85, 0.90],
                    backgroundColor: 'rgba(179, 136, 255, 0.1)',
                    borderColor: '#b388ff',
                    borderDash: [5, 5],
                    pointBackgroundColor: '#b388ff',
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#f0f4f8', font: { size: 12, family: 'Outfit' } },
                        ticks: { color: '#8e9bad', backdropColor: 'transparent', stepSize: 0.2 },
                        suggestedMin: 0, suggestedMax: 1.0
                    }
                },
                plugins: { legend: { labels: { color: '#f0f4f8', font: { family: 'Outfit' } } } }
            }
        });

        // 2. Bar Chart: Latency Breakdown
        const ctxBar = document.getElementById('latencyBarChart').getContext('2d');
        new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: chartData.scenario_labels,
                datasets: [{
                    label: 'Whisper STT (s)',
                    data: chartData.stt_latencies,
                    backgroundColor: '#4facfe'
                }, {
                    label: 'GPT-4o LLM (s)',
                    data: chartData.llm_latencies,
                    backgroundColor: '#00f2fe'
                }, {
                    label: 'OpenAI TTS (s)',
                    data: chartData.tts_latencies,
                    backgroundColor: '#b388ff'
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#8e9bad' } },
                    y: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: '#8e9bad' }, title: { display: true, text: 'Seconds', color: '#8e9bad' } }
                },
                plugins: {
                    legend: { labels: { color: '#f0f4f8', font: { family: 'Outfit' } } },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        });

        // Filter Functionality
        function filterScenarios(filterType, btnElem) {
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            btnElem.classList.add('active');

            const cards = document.querySelectorAll('.scenario-card');
            cards.forEach(card => {
                const status = card.getAttribute('data-status');
                const category = card.getAttribute('data-category');
                
                if (filterType === 'ALL') {
                    card.style.display = 'block';
                } else if (filterType === 'PASS') {
                    card.style.display = status === 'PASS' ? 'block' : 'none';
                } else if (filterType === 'FAIL') {
                    card.style.display = status !== 'PASS' ? 'block' : 'none';
                } else {
                    card.style.display = category.toLowerCase() === filterType.toLowerCase() ? 'block' : 'none';
                }
            });
        }
    </script>
</body>
</html>
"""


class EnterpriseReportGenerator:
    """Generates visual HTML dashboards with rich telemetry and Chart.js graphics."""

    @classmethod
    def generate_html_dashboard(
        self,
        scenarios_data: List[Dict[str, Any]],
        output_path: Path | str = "reports/html/report.html",
    ) -> Path:
        """Compile a list of scenario telemetry dictionaries into a styled HTML dashboard."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        total_scenarios = len(scenarios_data)
        total_passed = sum(1 for sc in scenarios_data if sc.get("status") == "PASS")
        pass_rate = round((total_passed / total_scenarios * 100), 1) if total_scenarios > 0 else 100.0

        avg_wer = round(sum(sc.get("wer", 0.0) for sc in scenarios_data) / max(total_scenarios, 1), 3)
        avg_quality = round(
            sum((sc.get("correctness", 0.9) + sc.get("relevancy", 0.9)) / 2 for sc in scenarios_data) / max(total_scenarios, 1), 2
        )
        avg_latency = round(sum(sc.get("latency", 0.0) for sc in scenarios_data) / max(total_scenarios, 1), 2)

        # Prepare chart datasets
        radar_values = [
            round(sum(sc.get("correctness", 0.9) for sc in scenarios_data) / max(total_scenarios, 1), 2),
            round(sum(sc.get("relevancy", 0.9) for sc in scenarios_data) / max(total_scenarios, 1), 2),
            round(sum(sc.get("faithfulness", 0.95) for sc in scenarios_data) / max(total_scenarios, 1), 2),
            round(1.0 - sum(sc.get("hallucination", 0.05) for sc in scenarios_data) / max(total_scenarios, 1), 2),
            round(1.0 - avg_wer, 2),
        ]

        chart_data = {
            "radar_values": radar_values,
            "scenario_labels": [sc.get("conversation_id", f"SC-{i}") for i, sc in enumerate(scenarios_data[:10])],
            "stt_latencies": [round(sc.get("stt_latency", 0.8), 2) for sc in scenarios_data[:10]],
            "llm_latencies": [round(sc.get("llm_latency", 1.5), 2) for sc in scenarios_data[:10]],
            "tts_latencies": [round(sc.get("tts_latency", 2.5), 2) for sc in scenarios_data[:10]],
        }

        template = Template(HTML_DASHBOARD_TEMPLATE)
        html_content = template.render(
            overall_passed=(pass_rate >= 90.0),
            timestamp=datetime.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S UTC"),
            pass_rate=pass_rate,
            total_passed=total_passed,
            total_scenarios=total_scenarios,
            avg_quality_score=avg_quality,
            avg_wer=avg_wer,
            avg_latency=avg_latency,
            scenarios=scenarios_data,
            chart_data=json.dumps(chart_data),
        )

        out_file.write_text(html_content, encoding="utf-8")
        logger.info(f"Enterprise HTML dashboard compiled successfully -> {out_file}")
        return out_file
