import pandas as pd
import streamlit as st
import sqlite3
from io import BytesIO
import zipfile
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet


# --- CSV to PDF function ---
def csvs_to_pdf(named_dfs, cols_per_page=8):
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    for file_name, df in named_dfs:
        # Title with file name
        elements.append(Paragraph(f"📄 {file_name}", styles["Heading2"]))

        # Break wide DataFrame into column chunks
        for start in range(0, len(df.columns), cols_per_page):
            chunk = df.iloc[:, start:start+cols_per_page]
            data = [chunk.columns.tolist()]

            for row in chunk.values.tolist():
                wrapped_row = [Paragraph(str(cell), styles["Normal"]) for cell in row]
                data.append(wrapped_row)

            table = Table(data, repeatRows=1)
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.whitesmoke])
            ])
            table.setStyle(style)

            elements.append(table)
            elements.append(PageBreak())

    pdf.build(elements)
    buffer.seek(0)
    return buffer


# --- DB to CSV ZIP function ---
def db_to_csv_zip(db_file):
    buffer = BytesIO()

    # Save uploaded file to a temporary file so sqlite3 can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(db_file.read())
        tmp_path = tmp.name

    with sqlite3.connect(tmp_path) as conn:
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for (table_name,) in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                zipf.writestr(f"{table_name}.csv", csv_bytes)

    buffer.seek(0)
    return buffer


# --- Streamlit UI ---
st.set_page_config(page_title="Data Converter", layout="wide")
st.title("📊 Data Converter App")

tab1, tab2 = st.tabs(["📄 CSV to PDF", "🗄️ DB to CSV (ZIP)"])

# --- Tab 1: CSV to PDF ---
with tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        named_dfs = [(file.name, pd.read_csv(file)) for file in uploaded_files]

        st.subheader("Preview of Uploaded CSVs")
        for file_name, df in named_dfs:
            st.markdown(f"**{file_name}**")
            st.dataframe(df.head(5))

        cols_per_page = st.sidebar.slider("Columns per page", 4, 12, 8)

        if st.button("Convert All to PDF"):
            pdf_buffer = csvs_to_pdf(named_dfs, cols_per_page=cols_per_page)
            st.download_button(
                label="⬇️ Download Combined PDF",
                data=pdf_buffer,
                file_name="combined_output.pdf",
                mime="application/pdf"
            )

# --- Tab 2: DB to CSV ZIP ---
with tab2:
    db_file = st.file_uploader("Upload a SQLite .db file", type=["db"])

    if db_file is not None:
        if st.button("Convert DB to CSV ZIP"):
            zip_buffer = db_to_csv_zip(db_file)
            st.download_button(
                label="⬇️ Download CSVs as ZIP",
                data=zip_buffer,
                file_name="db_tables.zip",
                mime="application/zip"
            )
