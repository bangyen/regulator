"""
Enhanced Monitoring Dashboard

This module provides comprehensive monitoring visualization with multiple metrics,
continuous risk scores, and dynamic monitoring patterns.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd


class EnhancedMonitoringDashboard:
    """
    Enhanced monitoring dashboard with multiple visualization options.

    Provides:
    1. Continuous risk score visualization
    2. Graduated penalty tracking
    3. Market volatility analysis
    4. Violation pattern analysis
    5. Comparative monitoring across different regulators
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the enhanced monitoring dashboard.

        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)
        self.colors = {
            "market_volatility": "#4ECDC4",
            "penalties": "#96CEB4",
            "severe_violations": "#FF9FF3",
            "moderate_violations": "#54A0FF",
            "minor_violations": "#5F27CD",
        }

    def load_episode_data(self, episode_file: str) -> Dict[str, Any]:
        """Load episode data from JSONL file."""
        episode_data: Dict[str, Any] = {"steps": []}
        episode_file_path = self.log_dir / episode_file

        if not episode_file_path.exists():
            raise FileNotFoundError(f"Episode file not found: {episode_file_path}")

        with open(episode_file_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("type") == "step":
                        episode_data["steps"].append(data)

        return episode_data

    def create_comprehensive_dashboard(
        self,
        episode_files: List[str],
        output_file: str = "enhanced_monitoring_dashboard.png",
    ) -> None:
        """
        Create a comprehensive monitoring dashboard.

        Args:
            episode_files: List of episode files to analyze
            output_file: Output file name for the dashboard
        """
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(
            "Enhanced Regulator Monitoring Dashboard", fontsize=20, fontweight="bold"
        )

        # Load and analyze all episodes
        episode_data: Dict[str, Any] = {}
        for episode_file in episode_files:
            try:
                data = self.load_episode_data(episode_file)
                episode_data[episode_file] = data
            except FileNotFoundError:
                print(f"Warning: Could not load {episode_file}")
                continue

        if not episode_data:
            print("No episode data found!")
            return

        # Create visualizations
        self._plot_fines_over_time(axes[0, 0], episode_data)
        self._plot_fines_over_time(axes[0, 1], episode_data)
        self._plot_market_volatility(axes[1, 0], episode_data)
        self._plot_penalty_analysis(axes[1, 1], episode_data)
        self._plot_violation_severity(axes[2, 0], episode_data)
        self._plot_monitoring_summary(axes[2, 1], episode_data)

        plt.tight_layout()
        plt.savefig(self.log_dir / output_file, dpi=300, bbox_inches="tight")
        print(f"Enhanced dashboard saved to: {self.log_dir / output_file}")

    def _plot_fines_over_time(self, ax: Any, episode_data: Dict[str, Any]) -> None:
        """Plot fines over time."""
        ax.set_title("Fines Over Time", fontweight="bold")

        for i, (episode_file, data) in enumerate(episode_data.items()):
            steps = data["steps"]
            if not steps:
                continue

            # Extract fines
            fines = []
            for step in steps:
                regulator_flags = step.get("regulator_flags", {})
                total_fine = sum(regulator_flags.get("fines_applied", []))
                fines.append(total_fine)

            if fines:
                steps_range = range(1, len(fines) + 1)
                ax.plot(
                    steps_range,
                    fines,
                    label=episode_file.replace(".jsonl", ""),
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Total Fines")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_market_volatility(self, ax: Any, episode_data: Dict[str, Any]) -> None:
        """Plot market volatility over time."""
        ax.set_title("Market Volatility Analysis", fontweight="bold")

        for i, (episode_file, data) in enumerate(episode_data.items()):
            steps = data["steps"]
            if not steps:
                continue

            # Extract market volatility (if available)
            volatility_scores = []
            for step in steps:
                volatility = step.get("regulator_flags", {}).get(
                    "market_volatility", 0.0
                )
                volatility_scores.append(volatility)

            if volatility_scores:
                steps_range = range(1, len(volatility_scores) + 1)
                ax.plot(
                    steps_range,
                    volatility_scores,
                    label=episode_file.replace(".jsonl", ""),
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Market Volatility")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add volatility threshold
        ax.axhline(
            y=0.3,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="High Volatility Threshold",
        )

    def _plot_penalty_analysis(self, ax: Any, episode_data: Dict[str, Any]) -> None:
        """Plot penalty analysis and cumulative fines."""
        ax.set_title("Penalty Analysis and Cumulative Fines", fontweight="bold")

        for i, (episode_file, data) in enumerate(episode_data.items()):
            steps = data["steps"]
            if not steps:
                continue

            # Extract penalty data
            fines_applied = []
            cumulative_fines = 0

            for step in steps:
                step_fines = step.get("regulator_flags", {}).get(
                    "fines_applied", [0, 0, 0]
                )
                if isinstance(step_fines, list):
                    step_total = sum(step_fines)
                else:
                    step_total = step_fines

                cumulative_fines += step_total
                fines_applied.append(cumulative_fines)

            if fines_applied:
                steps_range = range(1, len(fines_applied) + 1)
                ax.plot(
                    steps_range,
                    fines_applied,
                    label=episode_file.replace(".jsonl", ""),
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Fines")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_violation_severity(self, ax: Any, episode_data: Dict[str, Any]) -> None:
        """Plot violation severity distribution."""
        ax.set_title("Violation Severity Distribution", fontweight="bold")

        severity_counts = {"minor": 0, "moderate": 0, "severe": 0, "critical": 0}

        for episode_file, data in episode_data.items():
            steps = data["steps"]
            if not steps:
                continue

            # Extract severity data (if available)
            for step in steps:
                severities = step.get("regulator_flags", {}).get(
                    "violation_severities", []
                )
                for severity in severities:
                    if severity in severity_counts:
                        severity_counts[severity] += 1

        # Create bar chart
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = [
            self.colors["minor_violations"],
            self.colors["moderate_violations"],
            self.colors["severe_violations"],
            self.colors["severe_violations"],
        ]

        bars = ax.bar(severities, counts, color=colors, alpha=0.7)
        ax.set_xlabel("Violation Severity")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{count}",
                ha="center",
                va="bottom",
            )

    def _plot_monitoring_summary(self, ax: Any, episode_data: Dict[str, Any]) -> None:
        """Plot monitoring summary statistics."""
        ax.set_title("Monitoring Summary Statistics", fontweight="bold")

        # Calculate summary statistics
        summary_stats = []
        episode_names = []

        for episode_file, data in episode_data.items():
            steps = data["steps"]
            if not steps:
                continue

            # Calculate statistics
            parallel_violations = sum(
                step.get("regulator_flags", {}).get("parallel_violation", False)
                for step in steps
            )
            structural_break_violations = sum(
                step.get("regulator_flags", {}).get("structural_break_violation", False)
                for step in steps
            )
            total_violations = parallel_violations + structural_break_violations
            violation_rate = total_violations / len(steps) if steps else 0

            summary_stats.append(
                {
                    "Total Steps": len(steps),
                    "Parallel Violations": parallel_violations,
                    "Structural Break Violations": structural_break_violations,
                    "Total Violations": total_violations,
                    "Violation Rate": violation_rate,
                }
            )
            episode_names.append(episode_file.replace(".jsonl", ""))

        # Create summary table
        ax.axis("tight")
        ax.axis("off")

        if summary_stats:
            df = pd.DataFrame(summary_stats, index=episode_names)
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                rowLabels=df.index,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

    def generate_monitoring_report(
        self, episode_files: List[str], output_file: str = "monitoring_report.txt"
    ) -> None:
        """
        Generate a comprehensive text report of monitoring results.

        Args:
            episode_files: List of episode files to analyze
            output_file: Output report file name
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ENHANCED REGULATOR MONITORING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        for episode_file in episode_files:
            try:
                data = self.load_episode_data(episode_file)
                steps = data["steps"]

                if not steps:
                    continue

                report_lines.append(f"EPISODE: {episode_file}")
                report_lines.append("-" * 50)

                # Calculate statistics
                parallel_violations = sum(
                    step.get("regulator_flags", {}).get("parallel_violation", False)
                    for step in steps
                )
                structural_break_violations = sum(
                    step.get("regulator_flags", {}).get(
                        "structural_break_violation", False
                    )
                    for step in steps
                )
                total_violations = parallel_violations + structural_break_violations
                violation_rate = total_violations / len(steps) if steps else 0

                # Market volatility
                volatility_scores = [
                    step.get("regulator_flags", {}).get("market_volatility", 0.0)
                    for step in steps
                ]
                avg_volatility = (
                    np.mean(volatility_scores) if volatility_scores else 0.0
                )

                # Penalties
                total_fines = 0
                for step in steps:
                    fines = step.get("regulator_flags", {}).get(
                        "fines_applied", [0, 0, 0]
                    )
                    if isinstance(fines, list):
                        total_fines += sum(fines)
                    else:
                        total_fines += fines

                # Write statistics
                report_lines.append(f"Total Steps: {len(steps)}")
                report_lines.append(f"Parallel Violations: {parallel_violations}")
                report_lines.append(
                    f"Structural Break Violations: {structural_break_violations}"
                )
                report_lines.append(f"Total Violations: {total_violations}")
                report_lines.append(f"Violation Rate: {violation_rate:.1%}")
                report_lines.append(f"Average Market Volatility: {avg_volatility:.3f}")
                report_lines.append(f"Total Fines Applied: {total_fines:.2f}")
                report_lines.append("")

            except FileNotFoundError:
                report_lines.append(f"ERROR: Could not load {episode_file}")
                report_lines.append("")

        # Write report to file
        with open(self.log_dir / output_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Monitoring report saved to: {self.log_dir / output_file}")


def main() -> None:
    """Example usage of the enhanced monitoring dashboard."""
    dashboard = EnhancedMonitoringDashboard()

    # Example episode files (adjust as needed)
    episode_files = [
        "test_new_thresholds.jsonl",
        "test_ml_regulator.jsonl",
        "improved_regular_regulator.jsonl",
        "improved_ml_regulator.jsonl",
    ]

    # Create comprehensive dashboard
    dashboard.create_comprehensive_dashboard(episode_files)

    # Generate monitoring report
    dashboard.generate_monitoring_report(episode_files)


if __name__ == "__main__":
    main()
