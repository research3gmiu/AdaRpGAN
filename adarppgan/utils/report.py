"""
Results Report Generator for AdaRpGAN.

Generates a comprehensive experiment report in multiple formats:
  • HTML   — full visual report with embedded plots and tables
  • PDF    — self-contained printable report with embedded figures
  • CSV    — machine-readable results table
  • TXT    — plain-text summary for quick reference
  • JSON   — machine-readable full dump

Usage
-----
    from utils.report import generate_report
    generate_report(results_dict, output_dir="./results/report")
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Optional


def generate_report(
    results:    dict,
    output_dir: str = "./results/report",
    title:      str = "AdaRpGAN Experiment Report",
) -> dict[str, str]:
    """
    Generate experiment report in multiple formats.

    Parameters
    ----------
    results : dict
        Expected keys:
        - "conditions": list of dicts, each with:
            - "name": str (e.g., "AdaRpGAN (full)")
            - "fid": float
            - "is_mean": float (optional)
            - "is_std": float (optional)
            - "epochs": int
            - "time_seconds": float
            - "final_gamma": float (optional)
            - "final_ncritic": int (optional)
        - "dataset": str (e.g., "CIFAR-10")
        - "img_size": int
        - "device": str
        - "timestamp": str
        - "figures": dict[str, str] — name → path to figure file
        - "config": dict (optional) — training configuration
        - "sample_images": dict[str, str] (optional) — condition name → path

    output_dir : str
        Directory to save report files.

    Returns
    -------
    dict of format → filepath
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # CSV
    csv_path = os.path.join(output_dir, "results.csv")
    _write_csv(results, csv_path)
    paths["csv"] = csv_path

    # TXT
    txt_path = os.path.join(output_dir, "results.txt")
    _write_txt(results, txt_path, title)
    paths["txt"] = txt_path

    # HTML
    html_path = os.path.join(output_dir, "report.html")
    _write_html(results, html_path, title, output_dir)
    paths["html"] = html_path

    # JSON (machine-readable full dump)
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    paths["json"] = json_path

    # PDF (self-contained printable report)
    pdf_path = os.path.join(output_dir, "report.pdf")
    _write_pdf(results, pdf_path, title, output_dir)
    if os.path.exists(pdf_path):
        paths["pdf"] = pdf_path

    print(f"\n[Report] Generated reports:")
    for fmt, p in paths.items():
        print(f"  {fmt:5s} → {p}")

    return paths


def _write_csv(results: dict, path: str) -> None:
    conditions = results.get("conditions", [])
    if not conditions:
        return

    fieldnames = ["name", "fid", "is_mean", "is_std", "epochs",
                  "time_seconds", "final_gamma", "final_ncritic"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in conditions:
            writer.writerow(c)


def _write_txt(results: dict, path: str, title: str) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)
    lines.append(f"  Dataset:    {results.get('dataset', 'N/A')}")
    lines.append(f"  Resolution: {results.get('img_size', 'N/A')}×{results.get('img_size', 'N/A')}")
    lines.append(f"  Device:     {results.get('device', 'N/A')}")
    lines.append(f"  Timestamp:  {results.get('timestamp', 'N/A')}")
    lines.append("")

    conditions = results.get("conditions", [])
    if conditions:
        lines.append(f"  {'Condition':<35s} {'FID↓':>8s} {'IS↑':>10s} {'Time':>10s}")
        lines.append("  " + "-" * 65)
        for c in conditions:
            name = c.get("name", "?")
            fid  = c.get("fid", float("nan"))
            is_m = c.get("is_mean")
            is_s = c.get("is_std")
            secs = c.get("time_seconds", 0)

            is_str = f"{is_m:.2f}±{is_s:.2f}" if is_m is not None else "N/A"
            t_str  = _fmt_time(secs)
            lines.append(f"  {name:<35s} {fid:>8.2f} {is_str:>10s} {t_str:>10s}")

    lines.append("")
    lines.append("=" * 70)

    # Best result highlight
    if conditions:
        best = min(conditions, key=lambda c: c.get("fid", float("inf")))
        lines.append(f"  ★ Best FID: {best.get('fid', 'N/A'):.2f} ({best.get('name', '?')})")
        if len(conditions) >= 2:
            baseline = max(conditions, key=lambda c: c.get("fid", 0))
            improvement = (baseline["fid"] - best["fid"]) / baseline["fid"] * 100
            lines.append(f"  ★ Improvement over worst baseline: {improvement:.1f}%")
    lines.append("=" * 70)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_html(results: dict, path: str, title: str, output_dir: str) -> None:
    conditions = results.get("conditions", [])
    figures    = results.get("figures", {})
    samples    = results.get("sample_images", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d29;
    --text: #e4e4e7;
    --muted: #71717a;
    --accent: #6366f1;
    --green: #22c55e;
    --border: #27272a;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--accent), #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
  .meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .meta-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
  }}
  .meta-card .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .meta-card .value {{ font-size: 1.3rem; font-weight: 600; margin-top: 0.3rem; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
  }}
  th {{
    background: #1e1e2e;
    padding: 1rem;
    text-align: left;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }}
  td {{
    padding: 0.8rem 1rem;
    border-top: 1px solid var(--border);
  }}
  tr.best td {{ color: var(--green); font-weight: 600; }}
  .figures {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}
  .fig-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  .fig-card img {{ width: 100%; display: block; }}
  .fig-card .caption {{ padding: 0.8rem 1rem; color: var(--muted); font-size: 0.85rem; }}
  .highlight {{
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }}
  .highlight h3 {{ color: var(--accent); margin-bottom: 0.5rem; }}
  footer {{ color: var(--muted); font-size: 0.8rem; margin-top: 3rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p class="subtitle">Generated by AdaRpGAN Framework — {results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>

  <div class="meta">
    <div class="meta-card">
      <div class="label">Dataset</div>
      <div class="value">{results.get('dataset', 'N/A')}</div>
    </div>
    <div class="meta-card">
      <div class="label">Resolution</div>
      <div class="value">{results.get('img_size', '?')}×{results.get('img_size', '?')}</div>
    </div>
    <div class="meta-card">
      <div class="label">Device</div>
      <div class="value">{results.get('device', 'N/A')}</div>
    </div>
    <div class="meta-card">
      <div class="label">Conditions Tested</div>
      <div class="value">{len(conditions)}</div>
    </div>
  </div>
"""

    # Results table
    if conditions:
        best_fid = min(c.get("fid", float("inf")) for c in conditions)
        html += """
  <h2 style="margin-bottom:1rem;">Results</h2>
  <table>
    <thead>
      <tr><th>Condition</th><th>FID ↓</th><th>IS ↑</th><th>Epochs</th><th>Time</th><th>Final γ</th><th>n_critic</th></tr>
    </thead>
    <tbody>
"""
        for c in conditions:
            fid  = c.get("fid", float("nan"))
            is_m = c.get("is_mean")
            is_s = c.get("is_std")
            is_str = f"{is_m:.2f} ± {is_s:.2f}" if is_m is not None else "—"
            t_str  = _fmt_time(c.get("time_seconds", 0))
            gamma  = c.get("final_gamma")
            nc     = c.get("final_ncritic")
            gamma_str = f"{gamma:.2f}" if gamma is not None else "—"
            nc_str    = str(nc) if nc is not None else "—"
            row_class = ' class="best"' if abs(fid - best_fid) < 0.01 else ""

            html += f'      <tr{row_class}><td>{c.get("name","?")}</td>'
            html += f'<td>{fid:.2f}</td><td>{is_str}</td>'
            html += f'<td>{c.get("epochs","?")}</td><td>{t_str}</td>'
            html += f'<td>{gamma_str}</td><td>{nc_str}</td></tr>\n'

        html += "    </tbody>\n  </table>\n"

        # Highlight box
        best = min(conditions, key=lambda c: c.get("fid", float("inf")))
        html += f"""
  <div class="highlight">
    <h3>★ Best Result</h3>
    <p><strong>{best.get('name','?')}</strong> achieved FID = <strong>{best.get('fid', '?'):.2f}</strong></p>
"""
        if len(conditions) >= 2:
            worst = max(conditions, key=lambda c: c.get("fid", 0))
            imp = (worst["fid"] - best["fid"]) / worst["fid"] * 100
            html += f"    <p>Improvement over {worst.get('name','baseline')}: <strong>{imp:.1f}%</strong></p>\n"
        html += "  </div>\n"

    # Figures
    if figures:
        html += '  <h2 style="margin-bottom:1rem;">Figures</h2>\n  <div class="figures">\n'
        for name, fig_path in figures.items():
            rel_path = os.path.relpath(fig_path, output_dir) if os.path.isabs(fig_path) else fig_path
            html += f'    <div class="fig-card"><img src="{rel_path}" alt="{name}"><div class="caption">{name}</div></div>\n'
        html += "  </div>\n"

    # Sample images
    if samples:
        html += '  <h2 style="margin-bottom:1rem;">Generated Samples</h2>\n  <div class="figures">\n'
        for name, img_path in samples.items():
            rel_path = os.path.relpath(img_path, output_dir) if os.path.isabs(img_path) else img_path
            html += f'    <div class="fig-card"><img src="{rel_path}" alt="{name}"><div class="caption">{name}</div></div>\n'
        html += "  </div>\n"

    html += f"""
  <footer>
    AdaRpGAN Framework v1.0 — Self-Tuning GAN Training<br>
    Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </footer>
</div>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


# ──────────────────────────────────────────────
# PDF Report (fpdf2)
# ──────────────────────────────────────────────

def _write_pdf(results: dict, path: str, title: str, output_dir: str) -> None:
    """Generate a self-contained PDF report with embedded figures."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("  [Report] ⚠ fpdf2 not installed — skipping PDF. Install with: pip install fpdf2")
        return

    conditions = results.get("conditions", [])
    figures    = results.get("figures", {})
    samples    = results.get("sample_images", {})

    class PDFReport(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "AdaRpGAN Experiment Report", align="L")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = PDFReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────
    title_pdf = title.replace("—", "-")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(60, 60, 180)
    pdf.cell(0, 14, title_pdf, ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Generated: {results.get('timestamp', 'N/A')}", ln=True, align="C")
    pdf.ln(6)

    # ── Metadata row ──────────────────────────────────────────────
    meta_items = [
        ("Dataset", results.get("dataset", "N/A")),
        ("Resolution", f"{results.get('img_size', '?')}x{results.get('img_size', '?')}"),
        ("Device", results.get("device", "N/A")),
        ("Conditions", str(len(conditions))),
    ]
    col_w = (pdf.w - 20) / len(meta_items)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    for label, _ in meta_items:
        pdf.cell(col_w, 5, label.upper(), align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    for _, value in meta_items:
        pdf.cell(col_w, 7, str(value), align="C")
    pdf.ln(10)

    # ── Results table ─────────────────────────────────────────────
    if conditions:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Results", ln=True)
        pdf.ln(2)

        headers = ["Condition", "FID (lower)", "IS (higher)", "Epochs", "Time", "Final g"]
        col_widths = [55, 22, 28, 20, 20, 22]
        best_fid = min(c.get("fid", float("inf")) for c in conditions)

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(40, 40, 60)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, border=1, fill=True, align="C")
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 9)
        for i, c in enumerate(conditions):
            fid  = c.get("fid", float("nan"))
            is_m = c.get("is_mean")
            is_s = c.get("is_std")
            is_str = f"{is_m:.2f}+/-{is_s:.2f}" if is_m is not None else "-"
            t_str  = _fmt_time(c.get("time_seconds", 0))
            gamma  = c.get("final_gamma")
            g_str  = f"{gamma:.2f}" if gamma is not None else "-"

            is_best = abs(fid - best_fid) < 0.01
            if is_best:
                pdf.set_text_color(0, 130, 60)
                pdf.set_font("Helvetica", "B", 9)
            else:
                pdf.set_text_color(30, 30, 30)
                pdf.set_font("Helvetica", "", 9)

            bg = (245, 245, 250) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*bg)

            row_data = [c.get("name", "?"), f"{fid:.2f}", is_str,
                        str(c.get("epochs", "?")), t_str, g_str]
            for val, w in zip(row_data, col_widths):
                pdf.cell(w, 7, val, border=1, fill=True, align="C")
            pdf.ln()

        pdf.ln(4)

        # Highlight best result
        best = min(conditions, key=lambda c: c.get("fid", float("inf")))
        pdf.set_fill_color(230, 245, 235)
        pdf.set_text_color(0, 100, 50)
        pdf.set_font("Helvetica", "B", 11)
        highlight_text = f"* Best: {best.get('name', '?')} -- FID = {best.get('fid', '?'):.2f}"
        if len(conditions) >= 2:
            worst = max(conditions, key=lambda c: c.get("fid", 0))
            imp = (worst["fid"] - best["fid"]) / worst["fid"] * 100
            highlight_text += f"  ({imp:.1f}% improvement)"
        pdf.cell(0, 10, highlight_text, ln=True, fill=True, align="C")
        pdf.ln(6)

    # ── Figures ────────────────────────────────────────────────────
    all_images = {}
    all_images.update(figures)
    all_images.update(samples)

    if all_images:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Figures", ln=True)
        pdf.ln(2)

        for fig_name, fig_path in all_images.items():
            if not os.path.exists(fig_path):
                continue
            # Check if we need a new page (leave room for image + caption)
            if pdf.get_y() > 200:
                pdf.add_page()
            try:
                img_w = pdf.w - 30  # page width minus margins
                pdf.image(fig_path, x=15, w=img_w)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 6, fig_name, ln=True, align="C")
                pdf.ln(4)
            except Exception as e:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(180, 50, 50)
                pdf.cell(0, 6, f"[Could not embed: {fig_name} - {e}]", ln=True)
                pdf.ln(2)

    # ── Config details ────────────────────────────────────────────
    config = results.get("config", {})
    if config:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Training Configuration", ln=True)
        pdf.ln(2)

        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(50, 50, 50)
        for k, v in config.items():
            pdf.cell(50, 6, str(k), border=0)
            pdf.cell(0, 6, str(v), border=0, ln=True)

    # ── Save ──────────────────────────────────────────────────────
    try:
        pdf.output(path)
        print(f"  [Report] PDF saved → {path}")
    except Exception as e:
        print(f"  [Report] ⚠ PDF generation failed: {e}")
