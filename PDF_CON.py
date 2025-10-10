"""
# """""
# streamlit_app_ui_enhanced.py
"""
# """
# Data Converter App — Enhanced UI/UX
# - Polished layout, icons, and visual hierarchy
# - Clear steps for each flow with progress & statuses
# - File chips + per-file previews in expanders
# - Sticky sidebar with grouped settings and help tooltips
# - Non-blocking toasts, success banners, and error surfaces
# - Skips empty CSV files/tables with visible warnings

# Tip: Save this file as `streamlit_app.py` and run `streamlit run streamlit_app.py`.
# """

import os
import re
from io import BytesIO
import zipfile
import tempfile
import sqlite3
import time
import gc

import pandas as pd
import streamlit as st
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import (
    SimpleDocTemplate,
    TableStyle,
    Paragraph,
    PageBreak,
    LongTable,
    KeepInFrame,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# NEW: components to run safe client-side JS
import streamlit.components.v1 as components

# =========================
# Theming / Global Config
# =========================

st.set_page_config(
    page_title="Data Converter — CSV↔PDF & DB→CSV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Reset keys for file uploaders (for the "Clear uploads" button) ---
st.session_state.setdefault("uploader_key_csv", 0)
st.session_state.setdefault("uploader_key_db", 0)

# Lightweight CSS polish
st.markdown(
    """
    <style>
      /* App width and typography tweaks */
      .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
      h1, h2, h3 { letter-spacing: .2px; }
      /* Chips */
      .chip {display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .6rem; border-radius:999px; border:1px solid var(--accent-border,#eaecef); background:rgba(127,127,127,.08); font-size:.85rem;}
      .chip .dot {width:.5rem; height:.5rem; border-radius:50%; background:var(--dot,#6c6c6c);} 
      .chip.good {--dot:#16a34a;}
      .chip.warn {--dot:#f59e0b;}
      .chip.bad  {--dot:#ef4444;}
      /* Section cards */
      .section-card {border:1px solid rgba(100,100,100,.2); border-radius:16px; padding:1rem 1.2rem; background:rgba(127,127,127,.04);} 
      .muted {color:rgba(120,120,120,.95)}
      .file-badge {font-weight:600;}
      /* Download button row spacing */
      .dl-row {display:flex; gap:.75rem; align-items:center; flex-wrap:wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Styles for ReportLab
# =========================

def make_styles():
    styles = getSampleStyleSheet()
    if "Cell" not in styles:
        styles.add(ParagraphStyle(
            name="Cell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=7.5,
            spaceAfter=0,
            spaceBefore=0,
        ))
    if "HeaderCell" not in styles:
        styles.add(ParagraphStyle(
            name="HeaderCell",
            parent=styles["Heading5"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=9.5,
            spaceAfter=0,
            spaceBefore=0,
        ))
    return styles

STYLES = make_styles()

# =========================
# Helpers
# =========================

def safe_cell_text(text, max_lines=10, max_chars_per_line=80):
    text = "" if text is None else str(text)
    text = xml_escape(text)
    raw_lines = re.split(r"\r?\n", text)
    wrapped = []
    for ln in raw_lines:
        for i in range(0, len(ln), max_chars_per_line):
            wrapped.append(ln[i:i + max_chars_per_line])
            if len(wrapped) >= max_lines:
                wrapped.append("...")
                return "<br/>".join(wrapped)
    return "<br/>".join(wrapped) if wrapped else ""


def make_cell_flowable(html, max_lines=10, line_leading=7.5, force_fit=False):
    para = Paragraph(html, STYLES["Cell"])
    if not force_fit:
        return para
    max_height = (max_lines * line_leading) + 2
    return KeepInFrame(120, max_height, [para], mode="shrink")


def _safe_unlink(path: str, attempts: int = 10, base_delay: float = 0.1):
    """
    Windows-friendly file removal with retries to avoid WinError 32
    when handles linger briefly after closing SQLite/IO.
    """
    if not path:
        return
    for i in range(attempts):
        try:
            os.remove(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(base_delay * (i + 1))
        except FileNotFoundError:
            return
    try:
        os.remove(path)
    except Exception:
        pass

# =========================
# Core Converters
# =========================

def csvs_to_pdf(named_dfs, cols_per_page=8, autosize=False, max_lines=10, max_chars_per_line=80, force_fit=True):
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []

    valid_named_dfs = []
    for name, df in named_dfs:
        if df.empty:
            st.warning(f"⚠️ Skipping '{name}' — file has no data.")
            continue
        valid_named_dfs.append((name, df))

    if not valid_named_dfs:
        raise ValueError("All uploaded CSV files are empty. Nothing to export.")

    total_chunks = sum(max(1, (len(df.columns) + cols_per_page - 1) // cols_per_page)
                       for _, df in valid_named_dfs)
    done = 0

    for file_index, (file_name, df) in enumerate(valid_named_dfs):
        elements.append(Paragraph(f"📄 {xml_escape(str(file_name))}", STYLES["Heading2"]))
        ncols = len(df.columns)
        for chunk_index, start in enumerate(range(0, ncols, cols_per_page)):
            chunk = df.iloc[:, start:start + cols_per_page]
            header_row = [Paragraph(xml_escape(str(col)), STYLES["HeaderCell"]) for col in chunk.columns]
            data = [header_row]

            for row in chunk.itertuples(index=False, name=None):
                wrapped_row = [
                    make_cell_flowable(
                        safe_cell_text(cell, max_lines=max_lines, max_chars_per_line=max_chars_per_line),
                        max_lines=max_lines,
                        line_leading=STYLES["Cell"].leading,
                        force_fit=force_fit,
                    ) for cell in row
                ]
                data.append(wrapped_row)

            table = LongTable(data, repeatRows=1)

            if autosize:
                def col_pt(series, header):
                    sample = series.astype(str).head(100).tolist()
                    longest = max([len(header)] + [len(s) for s in sample])
                    return max(40, min(220, longest * 3.0))
                widths = [col_pt(chunk[col], str(col)) for col in chunk.columns]
                table._argW = widths

            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.whitesmoke]),
            ])
            table.setStyle(style)

            elements.append(table)

            last_chunk = (start + cols_per_page) >= ncols
            last_file = (file_index == len(valid_named_dfs) - 1)
            if not (last_chunk and last_file):
                elements.append(PageBreak())

            done += 1
            # Progress callback is optional; left disabled by default.
            # st.session_state.get("csv_pdf_progress_cb", lambda *_: None)(file_name, chunk_index + 1, done, total_chunks)

    pdf.build(elements)
    buffer.seek(0)
    return buffer


def db_to_csv_zip(db_file, csv_chunk_rows=100_000):
    zip_buffer = BytesIO()
    tmp_db = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(db_file.read())
            tmp_db = tmp.name

        with sqlite3.connect(tmp_db) as conn:
            conn.text_factory = lambda b: b.decode(errors="replace") if isinstance(b, (bytes, bytearray)) else b
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for (table_name,) in tables:
                    # Quick existence check
                    probe = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 1', conn)
                    if probe.empty:
                        st.info(f"ℹ️ Skipping table '{table_name}' — no rows found.")
                        continue

                    offset = 0
                    first = True
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                        csv_path = tmp_csv.name
                    try:
                        while True:
                            chunk = pd.read_sql_query(
                                f'SELECT * FROM "{table_name}" LIMIT ? OFFSET ?',
                                conn, params=(csv_chunk_rows, offset))
                            # ensure the connection is not referenced by pandas objects
                            if chunk.empty:
                                break
                            chunk.to_csv(csv_path, mode="a", index=False, header=first, encoding="utf-8")
                            first = False
                            offset += len(chunk)
                        zipf.write(csv_path, arcname=f"{table_name}.csv")
                    finally:
                        if os.path.exists(csv_path):
                            _safe_unlink(csv_path)
    finally:
        # Close & unlink DB copy with Windows-safe retries
        if tmp_db and os.path.exists(tmp_db):
            _safe_unlink(tmp_db)

    zip_buffer.seek(0)
    return zip_buffer


# =========================
# UI — Header
# =========================

st.title("Data Converter App")
st.caption("Convert **CSV → PDF** and **SQLite DB → ZIP of CSVs** with robust wrapping, chunking, and progress.")


c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown('<span class="chip good"><span class="dot"></span> Stable PDF layout</span>', unsafe_allow_html=True)
with c2:
    st.markdown('<span class="chip good"><span class="dot"></span> Large-data ready</span>', unsafe_allow_html=True)
with c3:
    st.markdown('<span class="chip warn"><span class="dot"></span> Auto-skip empties</span>', unsafe_allow_html=True)

st.divider()

# =========================
# Sidebar — Settings & Help (+ Clear uploads)
# =========================

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**PDF Table Layout**")
    cols_per_page = st.slider("Columns per page", 4, 12, 8, help="How many columns to place on each PDF page.")
    autosize = st.toggle("Auto-size column widths", value=False, help="Estimate column widths from sample data (slower).")

    st.markdown("**Cell Wrapping**")
    max_lines = st.slider("Max lines per cell", 4, 20, 10, help="Hard cap on wrapped lines inside each cell.")
    max_chars = st.slider("Max chars per line", 30, 200, 80, step=5, help="Fixed-width wrap. Lower values wrap earlier.")
    force_fit = st.toggle("Force-fit cells (shrink)", value=True, help="Prevents rows from exceeding page height.")

    st.markdown("**DB Export**")
    rows_per_chunk = st.number_input("CSV rows per chunk", min_value=10_000, max_value=1_000_000, step=10_000, value=100_000,
                                     help="Bigger chunks are faster but use more memory.")

    st.divider()
    with st.expander("❓ Tips"):
        st.markdown(
            "- If a PDF fails due to extreme content, lower **Max lines per cell** or keep **Force-fit** on.\n"
            "- Use **Auto-size** if columns look cramped.\n"
            "- Huge DB tables? Lower **rows per chunk**."
        )

    st.divider()
    # --- Clear uploads button: resets both CSV and DB uploaders ---
    # inside the "Clear uploads" button handler in the sidebar
    if st.button("🧹 Clear uploads", help="Remove uploaded files and previews from this session."):
        st.session_state["uploader_key_csv"] += 1
        st.session_state["uploader_key_db"] += 1
        st.session_state.pop("_scroll_to_downloads", None)

        # version-safe rerun
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()


# # Persist a progress callback for CSV→PDF
# prog = st.progress(0.0, text="Waiting for files…")
# msg = st.empty()

# def _cb(fname, chunk_idx, done, total):
#     msg.write(f"Processing **{fname}** · chunk {chunk_idx} · {done}/{total}")
#     prog.progress(min(done / total, 1.0), text=f"Building PDF… {done}/{total} chunks")

# st.session_state["csv_pdf_progress_cb"] = _cb

# =========================
# Tabs — Flows
# =========================

tab1, tab2 = st.tabs(["📄 CSV → PDF", "🗄️ SQLite DB → CSV (ZIP)"])

# ---------- Tab 1: CSV → PDF ----------
with tab1:
    st.subheader("CSV to PDF")
    st.markdown("Upload one or more CSV files. Empty files will be skipped automatically.")

    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="You can drop many files at once.",
        key=f"csv_uploader_{st.session_state['uploader_key_csv']}",  # <-- key for reset
    )

    if uploaded_files:
        named_dfs = []
        st.markdown("### Files")
        for f in uploaded_files:
            try:
                df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip", low_memory=False)
                named_dfs.append((f.name, df))
            except Exception:
                st.error(f"❌ Failed to read {f.name}")
                continue

            # File card with quick stats
            rows, cols = df.shape
            badge = "<span class='file-badge'>Empty</span>" if df.empty else f"<span class='file-badge'>{rows:,} × {cols:,}</span>"
            chip_class = "warn" if df.empty else "good"
            st.markdown(
                f"<div class='section-card'>📄 <strong>{f.name}</strong> &nbsp; "
                f"<span class='chip {chip_class}'><span class='dot'></span> {badge}</span></div>",
                unsafe_allow_html=True,
            )
            with st.expander(f"Preview — {f.name}"):
                if df.empty:
                    st.warning("(No data — will be skipped)")
                else:
                    st.dataframe(df.head(10))

        st.markdown("### Convert")
        cta_col1, cta_col2 = st.columns([1, 3])
        with cta_col1:
            convert = st.button("♻️ Convert all to PDF", type="primary", use_container_width=True)
        with cta_col2:
            st.caption("We’ll wrap and cap long cells, and paginate wide tables.")

        if convert:
            try:
                with st.status("Generating PDF…", expanded=False) as status:
                    pdf_buffer = csvs_to_pdf(
                        named_dfs,
                        cols_per_page=cols_per_page,
                        autosize=autosize,
                        max_lines=max_lines,
                        max_chars_per_line=max_chars,
                        force_fit=force_fit,
                    )
                    status.update(label="PDF built", state="complete")
                st.success("✅ PDF ready!", icon="✅")
                st.markdown('<div class="dl-row">', unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Download Combined PDF",
                    data=pdf_buffer,
                    file_name="combined_output.pdf",
                    mime="application/pdf",
                )
                st.markdown('</div>', unsafe_allow_html=True)
                st.toast("Combined PDF is ready.")
                # Auto-jump after successful PDF build
                st.session_state["_scroll_to_downloads"] = True
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

# ---------- Tab 2: DB → CSV (ZIP) ----------
with tab2:
    st.subheader("SQLite DB to CSV (ZIP)")
    st.markdown("Upload a `.db` file. Empty tables are skipped; large tables export in chunks.")

    # Multi-DB upload
    db_files = st.file_uploader(
        "Upload a SQLite .db file",
        type=["db"],
        help="Drag & drop a SQLite database file",
        accept_multiple_files=True,
        key=f"db_uploader_{st.session_state['uploader_key_db']}",  # <-- key for reset
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        run_db = st.button("📦 Convert DB to CSV ZIP", use_container_width=True)
    with c2:
        st.caption("Exports each non-empty table into a CSV inside a ZIP archive.")

    if db_files and run_db:
        try:
            with st.status("Converting DB…", expanded=False) as status:
                # Build a master ZIP that contains subfolders per DB
                master_zip = BytesIO()
                with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as master:
                    for f in db_files:
                        # Use existing single-DB converter
                        per_zip = db_to_csv_zip(f, csv_chunk_rows=int(rows_per_chunk))
                        db_base = os.path.splitext(os.path.basename(f.name))[0]
                        # Merge per-DB zip into master under db_base/
                        with zipfile.ZipFile(per_zip, "r") as zf_in:
                            for member in zf_in.namelist():
                                data = zf_in.read(member)
                                master.writestr(f"{db_base}/{member}", data)

                master_zip.seek(0)
                status.update(label="ZIP ready", state="complete")
            st.success("✅ Export complete!", icon="✅")
            st.markdown('<div class="dl-row">', unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download CSVs as ZIP",
                data=master_zip,
                file_name="db_tables.zip",
                mime="application/zip",
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.toast("ZIP exported successfully.")
            # Optional: jump after DB export as well
            st.session_state["_scroll_to_downloads"] = True
        except Exception as e:
            st.error(f"DB export failed: {e}")

# ============
# SCROLLER: render at end so the page is fully laid out when we scroll
# ============
if st.session_state.get("_scroll_to_downloads"):
    components.html(
        """
        <script>
            (function(){
                // Try bottom first
                window.parent.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                // Fallback: if a specific downloads container exists, scroll to it
                const anchors = Array.from(document.querySelectorAll('.dl-row'));
                if (anchors.length) {
                    anchors[anchors.length-1].scrollIntoView({behavior:'smooth', block:'end'});
                }
            })();
        </script>
        """,
        height=0,
    )
    # Reset flag so it doesn't keep auto-scrolling on next rerun
    st.session_state["_scroll_to_downloads"] = False
