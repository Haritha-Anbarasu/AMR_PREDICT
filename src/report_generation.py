"""
Module 6 (support): Generates a downloadable CSV and HTML summary report
from the final pipeline results.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_csv_report(results_df: pd.DataFrame, output_path: str) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path)
    return str(output_path)


def save_html_report(results_df: pd.DataFrame, genome_qc: dict, output_path: str) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    known = (results_df["final_category"] == "Confirmed ARG").sum()
    potential = (results_df["final_category"] == "Potential ARG").sum()
    non_arg = (results_df["final_category"] == "Non-ARG-like sequence").sum()

    html = f"""
    <html><head><title>AMR-PREDICT Report</title></head>
    <body style="font-family: sans-serif; max-width: 900px; margin: auto;">
        <h1>AMR-PREDICT Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <h2>Genome Summary</h2>
        <ul>
            <li>Genome size: {genome_qc.get('genome_size_mb')} Mb</li>
            <li>GC content: {genome_qc.get('gc_content_pct')}%</li>
            <li>Contigs: {genome_qc.get('num_contigs')}</li>
            <li>Total predicted genes: {len(results_df)}</li>
        </ul>
        <h2>ARG Summary</h2>
        <ul>
            <li>Confirmed ARGs: {known}</li>
            <li>Potential ARGs: {potential}</li>
            <li>Non-ARG-like genes: {non_arg}</li>
        </ul>
        <h2>Full Results</h2>
        {results_df.to_html()}
    </body></html>
    """
    output_path.write_text(html)
    return str(output_path)


if __name__ == "__main__":
    print("Import this module: save_csv_report(df, path), save_html_report(df, qc_dict, path)")
