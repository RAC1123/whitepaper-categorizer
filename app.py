import sys

# Ensure terminal output can handle unicode safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import io
import json
import sqlite3
import time
from datetime import datetime

import pdfplumber
import requests
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    send_file,
)
from openai import OpenAI
from werkzeug.utils import secure_filename

# ---------------------- CONFIG ----------------------

DB_PATH = "whitepapers.db"
UPLOAD_FOLDER = "uploads"
OPENAI_MODEL = "gpt-4o-mini"

INDUSTRIES = [
    "Banking", "Insurance", "Asset Management", "Hedge Funds", "Pension Funds",
    "Energy", "Technology", "Healthcare", "Industrials", "Consumer", "Real Estate",
    "Telecom", "Utilities", "Other"
]

MAIN_CATEGORIES = ["Retail", "Institutional", "Nondescript"]

# ---------------------- HELPER FUNCTIONS ----------------------

def ensure_upload_folder():
    """Make sure uploads folder exists."""
    if not os.path.isdir(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_db():
    """Create the SQLite database and whitepapers table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS whitepapers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            source TEXT,
            main_category TEXT,
            industry TEXT,
            short_summary TEXT,
            file_path TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()

def save_whitepaper(title, source, main_category, industry, short_summary, file_path=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO whitepapers (title, source, main_category, industry, short_summary, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (title, source, main_category, industry, short_summary, file_path, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def delete_whitepaper(wp_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Remove file from disk if we have one
    cur.execute("SELECT file_path FROM whitepapers WHERE id = ?;", (wp_id,))
    row = cur.fetchone()
    if row and row[0]:
        try:
            if os.path.isfile(row[0]):
                os.remove(row[0])
        except OSError:
            # Don't crash if file is missing
            pass
    cur.execute("DELETE FROM whitepapers WHERE id = ?;", (wp_id,))
    conn.commit()
    conn.close()

def load_whitepapers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, source, main_category, industry, short_summary, file_path, created_at
        FROM whitepapers;
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "title": r[1],
            "source": r[2],
            "main_category": r[3],
            "industry": r[4],
            "short_summary": r[5],
            "file_path": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]

def extract_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def classify(api_key: str, text: str) -> dict:
    client = OpenAI(api_key=api_key)
    snippet = text[:8000]

    system_prompt = (
        "You classify financial whitepapers by intended audience and industry and provide a short summary.\n\n"
        "Your primary task is to determine whether the document is intended for:\n"
        "- Institutional investors\n"
        "- Retail (general public) investors\n"
        "- Or if the audience is truly unclear (Nondescript).\n\n"
        "DEFINITIONS (INTENDED AUDIENCE):\n"
        "Institutional:\n"
        "- Intended for professional or sophisticated investors: pension funds, hedge funds, endowments, "
        "foundations, banks, insurance companies, asset managers, institutional consultants, sovereign wealth funds, "
        "corporate treasurers, etc.\n"
        "- Common signals: fiduciary duty, funding ratios, ALM, risk budgeting, mandate guidelines, RFQs/RFPs, "
        "benchmarks relative to institutional indices, regulatory capital, Solvency II, Basel, UCITS, complex "
        "derivatives, factor models, tracking error, ex-ante risk, multi-asset portfolio construction, "
        "liability-driven investing, overlays, optimization.\n"
        "- Writing style: assumes strong finance knowledge, heavy jargon, equations, quantitative methods, "
        "regulatory or policy detail, references to institutional asset allocation frameworks.\n\n"
        "Retail:\n"
        "- Intended for the general public or non-professional investors.\n"
        "- Common signals: basic investor education, explainers of what a bond/ETF/stock is, budgeting, saving, "
        "retirement basics, high-level product overviews, marketing to individuals, examples using everyday situations.\n"
        "- Writing style: plain language, defines basic terms, focuses on concepts like diversification and risk "
        "without deep math or institutional jargon.\n\n"
        "Nondescript:\n"
        "- Use ONLY when the audience is genuinely ambiguous and cannot reasonably be inferred.\n"
        "- Do NOT overuse this; if the language is clearly technical and finance-heavy, prefer Institutional.\n\n"
        "INDUSTRY:\n"
        "Choose one primary industry from this list only:\n"
        + ", ".join(INDUSTRIES) + "\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY):\n"
        "Respond ONLY with valid JSON in this exact structure, no extra text:\n"
        "{\n"
        "  \"title\": \"short inferred title\",\n"
        "  \"audience\": \"Institutional\" | \"Retail\" | \"Nondescript\",\n"
        "  \"audience_confidence\": integer 0-100,\n"
        "  \"audience_rationale\": \"one or two sentences explaining why\",\n"
        "  \"industry\": \"one industry from the list\",\n"
        "  \"short_summary\": \"2-3 sentence summary aimed at a financially literate reader\"\n"
        "}\n"
        "Do NOT include comments, trailing commas, or any text before or after the JSON."
    )

    user_prompt = (
        "Classify and summarize the following financial whitepaper text.\n\n"
        "Remember:\n"
        "- audience must be based on intended user: institutional vs retail vs nondescript.\n"
        "- industry must be one of the allowed industries.\n"
        "- audience_confidence is your confidence (0-100) in your audience choice.\n"
        "- audience_rationale should concisely explain the reasoning.\n\n"
        "Text:\n" + snippet
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    print("RAW MODEL RESPONSE:", repr(content))

    json_str = content
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = content[first_brace:last_brace + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Model did not return valid JSON.\n\n"
            "Raw reply was:\n"
            f"{content}\n\n"
            f"JSON error: {e}"
        )

    title = data.get("title", "Untitled").strip()
    audience = data.get("audience", "Nondescript").strip()
    industry = data.get("industry", "Other").strip()
    short_summary = data.get("short_summary", "").strip()

    if audience not in MAIN_CATEGORIES:
        main_category = "Nondescript"
    else:
        main_category = audience

    if industry not in INDUSTRIES:
        industry = "Other"

    return {
        "title": title,
        "main_category": main_category,
        "industry": industry,
        "short_summary": short_summary,
        "audience": audience,
        "audience_confidence": data.get("audience_confidence", 0),
        "audience_rationale": data.get("audience_rationale", "").strip(),
    }

# ---------------------- FLASK APP SETUP ----------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# On import (both locally and on Render), set up folders and DB
ensure_upload_folder()
init_db()


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>AI Whitepaper Categorizer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            margin: 0;
            background-color: #f4f6f8;
            color: #222;
        }
        .page {
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px 16px 40px;
        }
        h1 {
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #555;
            margin-bottom: 1.5rem;
        }
        .card {
            background-color: #fff;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .card h2 {
            margin-top: 0;
            font-size: 1.1rem;
        }
        label {
            display: block;
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        input[type=text],
        input[type=password],
        select {
            width: 100%;
            max-width: 420px;
            padding: 6px 8px;
            margin-top: 4px;
            border-radius: 4px;
            border: 1px solid #ccc;
            font-size: 0.9rem;
        }
        input[type=file] {
            margin-top: 6px;
        }
        button {
            margin-top: 10px;
            padding: 7px 14px;
            font-size: 0.9rem;
            border-radius: 4px;
            border: none;
            background-color: #2563eb;
            color: #fff;
            cursor: pointer;
        }
        button:hover {
            background-color: #1d4ed8;
        }
        button.delete {
            background-color: #b91c1c;
        }
        button.delete:hover {
            background-color: #991b1b;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        .error {
            background-color: #fee2e2;
            border: 1px solid #fecaca;
        }
        .success {
            background-color: #dcfce7;
            border: 1px solid #bbf7d0;
        }
        .filters-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: flex-end;
        }
        .filters-row label {
            margin-top: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 8px;
            font-size: 0.9rem;
            background-color: #fff;
        }
        th, td {
            border: 1px solid #e5e7eb;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background-color: #f9fafb;
            font-weight: 600;
        }
        .small-text {
            font-size: 0.8rem;
            color: #666;
        }
        .actions-col {
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <div class="page">
        <h1>AI Whitepaper Categorizer</h1>
        <div class="subtitle">
            Upload a financial whitepaper (PDF) via file or URL. The app will classify it,
            tag it by industry, and store it in a searchable library.
        </div>

        {% if message %}
            <div class="message {{ 'error' if error else 'success' }}">{{ message }}</div>
        {% endif %}

        <div class="card">
            <h2>Upload and Classify</h2>
            <form method="post" enctype="multipart/form-data">
                <input type="hidden" name="action" value="upload">
                <label>
                    OpenAI API Key:
                    <input type="password" name="api_key" required>
                </label>

                <label>
                    PDF File:
                    <input type="file" name="pdf_file" accept="application/pdf">
                </label>

                <div style="margin: 8px 0; font-size: 0.85rem; color: #555;">Or provide a PDF URL:</div>

                <label>
                    PDF URL:
                    <input type="text" name="pdf_url" placeholder="https://example.com/file.pdf">
                </label>

                <button type="submit">Process Whitepaper</button>
            </form>
        </div>

        <div class="card">
            <h2>Filter & Sort Library</h2>
            <form method="get">
                <div class="filters-row">
                    <label>
                        Main Category
                        <select name="filter_main_category">
                            <option value="">(All)</option>
                            {% for cat in main_categories %}
                                <option value="{{ cat }}" {% if cat == filter_main_category %}selected{% endif %}>{{ cat }}</option>
                            {% endfor %}
                        </select>
                    </label>

                    <label>
                        Industry
                        <select name="filter_industry">
                            <option value="">(All)</option>
                            {% for ind in industries %}
                                <option value="{{ ind }}" {% if ind == filter_industry %}selected{% endif %}>{{ ind }}</option>
                            {% endfor %}
                        </select>
                    </label>

                    <label>
                        Sort by date
                        <select name="sort_order">
                            <option value="newest" {% if sort_order == 'newest' %}selected{% endif %}>Newest first</option>
                            <option value="oldest" {% if sort_order == 'oldest' %}selected{% endif %}>Oldest first</option>
                        </select>
                    </label>

                    <div>
                        <button type="submit">Apply</button>
                    </div>
                </div>
            </form>
        </div>

        <div class="card">
            <h2>Library</h2>
            {% if whitepapers %}
                <table>
                    <tr>
                        <th>No.</th>
                        <th>Title</th>
                        <th>Main Category</th>
                        <th>Industry</th>
                        <th>Source</th>
                        <th>Short Summary</th>
                        <th>Created At (UTC)</th>
                        <th>Actions</th>
                    </tr>
                    {% for wp in whitepapers %}
                    <tr>
                        <!-- Dynamic numbering: loop.index is 1,2,3… regardless of DB id -->
                        <td>{{ loop.index }}</td>
                        <td>{{ wp["title"] }}</td>
                        <td>{{ wp["main_category"] }}</td>
                        <td>{{ wp["industry"] }}</td>
                        <td>
                            {% if wp["source"].startswith("URL: ") %}
                                <a href="{{ wp['source'][5:] }}" target="_blank">Open Source</a>
                            {% elif wp["file_path"] %}
                                <a href="{{ url_for('download_whitepaper', wp_id=wp['id']) }}">Download PDF</a>
                            {% else %}
                                <span class="small-text">{{ wp["source"] }}</span>
                            {% endif %}
                        </td>
                        <td>{{ wp["short_summary"] }}</td>
                        <td class="small-text">{{ wp["created_at"] }}</td>
                        <td class="actions-col">
                            <form method="post" style="display:inline;" onsubmit="return confirm('Delete this whitepaper?');">
                                <input type="hidden" name="action" value="delete">
                                <input type="hidden" name="delete_id" value="{{ wp['id'] }}">
                                <button type="submit" class="delete">Delete</button>
                            </form>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            {% else %}
                <p>No whitepapers saved yet. Upload one above.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

def ensure_upload_folder():
    if not os.path.isdir(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # New schema includes file_path for local downloads
    cur.execute("""
        CREATE TABLE IF NOT EXISTS whitepapers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            source TEXT,
            main_category TEXT,
            industry TEXT,
            short_summary TEXT,
            file_path TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()

def save_whitepaper(title, source, main_category, industry, short_summary, file_path=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO whitepapers (title, source, main_category, industry, short_summary, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (title, source, main_category, industry, short_summary, file_path, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def delete_whitepaper(wp_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Remove file from disk if we have one
    cur.execute("SELECT file_path FROM whitepapers WHERE id = ?;", (wp_id,))
    row = cur.fetchone()
    if row and row[0]:
        try:
            if os.path.isfile(row[0]):
                os.remove(row[0])
        except OSError:
            pass
    cur.execute("DELETE FROM whitepapers WHERE id = ?;", (wp_id,))
    conn.commit()
    conn.close()

def load_whitepapers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, source, main_category, industry, short_summary, file_path, created_at
        FROM whitepapers;
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "title": r[1],
            "source": r[2],
            "main_category": r[3],
            "industry": r[4],
            "short_summary": r[5],
            "file_path": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]

def extract_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def classify(api_key: str, text: str) -> dict:
    client = OpenAI(api_key=api_key)
    snippet = text[:8000]

    system_prompt = (
        "You classify financial whitepapers by intended audience and industry and provide a short summary.\n\n"
        "Your primary task is to determine whether the document is intended for:\n"
        "- Institutional investors\n"
        "- Retail (general public) investors\n"
        "- Or if the audience is truly unclear (Nondescript).\n\n"
        "DEFINITIONS (INTENDED AUDIENCE):\n"
        "Institutional:\n"
        "- Intended for professional or sophisticated investors: pension funds, hedge funds, endowments, "
        "foundations, banks, insurance companies, asset managers, institutional consultants, sovereign wealth funds, "
        "corporate treasurers, etc.\n"
        "- Common signals: fiduciary duty, funding ratios, ALM, risk budgeting, mandate guidelines, RFQs/RFPs, "
        "benchmarks relative to institutional indices, regulatory capital, Solvency II, Basel, UCITS, complex "
        "derivatives, factor models, tracking error, ex-ante risk, multi-asset portfolio construction, "
        "liability-driven investing, overlays, optimization.\n"
        "- Writing style: assumes strong finance knowledge, heavy jargon, equations, quantitative methods, "
        "regulatory or policy detail, references to institutional asset allocation frameworks.\n\n"
        "Retail:\n"
        "- Intended for the general public or non-professional investors.\n"
        "- Common signals: basic investor education, explainers of what a bond/ETF/stock is, budgeting, saving, "
        "retirement basics, high-level product overviews, marketing to individuals, examples using everyday situations.\n"
        "- Writing style: plain language, defines basic terms, focuses on concepts like diversification and risk "
        "without deep math or institutional jargon.\n\n"
        "Nondescript:\n"
        "- Use ONLY when the audience is genuinely ambiguous and cannot reasonably be inferred.\n"
        "- Do NOT overuse this; if the language is clearly technical and finance-heavy, prefer Institutional.\n\n"
        "INDUSTRY:\n"
        "Choose one primary industry from this list only:\n"
        + ", ".join(INDUSTRIES) + "\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY):\n"
        "Respond ONLY with valid JSON in this exact structure, no extra text:\n"
        "{\n"
        "  \"title\": \"short inferred title\",\n"
        "  \"audience\": \"Institutional\" | \"Retail\" | \"Nondescript\",\n"
        "  \"audience_confidence\": integer 0-100,\n"
        "  \"audience_rationale\": \"one or two sentences explaining why\",\n"
        "  \"industry\": \"one industry from the list\",\n"
        "  \"short_summary\": \"2-3 sentence summary aimed at a financially literate reader\"\n"
        "}\n"
        "Do NOT include comments, trailing commas, or any text before or after the JSON."
    )

    user_prompt = (
        "Classify and summarize the following financial whitepaper text.\n\n"
        "Remember:\n"
        "- audience must be based on intended user: institutional vs retail vs nondescript.\n"
        "- industry must be one of the allowed industries.\n"
        "- audience_confidence is your confidence (0-100) in your audience choice.\n"
        "- audience_rationale should concisely explain the reasoning.\n\n"
        "Text:\n" + snippet
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    print("RAW MODEL RESPONSE:", repr(content))

    json_str = content
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = content[first_brace:last_brace + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Model did not return valid JSON.\n\n"
            "Raw reply was:\n"
            f"{content}\n\n"
            f"JSON error: {e}"
        )

    title = data.get("title", "Untitled").strip()
    audience = data.get("audience", "Nondescript").strip()
    industry = data.get("industry", "Other").strip()
    short_summary = data.get("short_summary", "").strip()

    if audience not in MAIN_CATEGORIES:
        main_category = "Nondescript"
    else:
        main_category = audience

    if industry not in INDUSTRIES:
        industry = "Other"

    return {
        "title": title,
        "main_category": main_category,
        "industry": industry,
        "short_summary": short_summary,
        "audience": audience,
        "audience_confidence": data.get("audience_confidence", 0),
        "audience_rationale": data.get("audience_rationale", "").strip(),
    }

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    error = False

    filter_main_category = request.args.get("filter_main_category", "").strip()
    filter_industry = request.args.get("filter_industry", "").strip()
    sort_order = request.args.get("sort_order", "newest").strip() or "newest"

    if request.method == "POST":
        action = request.form.get("action", "upload")
        if action == "delete":
            try:
                delete_id = int(request.form.get("delete_id", "0"))
                delete_whitepaper(delete_id)
                return redirect(url_for("index",
                                        filter_main_category=filter_main_category,
                                        filter_industry=filter_industry,
                                        sort_order=sort_order))
            except Exception as e:
                message = f"Error deleting whitepaper: {e}"
                error = True
        else:
            try:
                api_key = request.form.get("api_key", "").strip()
                pdf_file = request.files.get("pdf_file")
                pdf_url = request.form.get("pdf_url", "").strip()

                if not api_key:
                    message = "OpenAI API key is required."
                    error = True
                elif not pdf_file and not pdf_url:
                    message = "Please upload a PDF file or provide a PDF URL."
                    error = True
                else:
                    file_path = None
                    if pdf_file and pdf_file.filename:
                        ensure_upload_folder()
                        safe_name = secure_filename(pdf_file.filename)
                        unique_name = f"{int(time.time())}_{safe_name}"
                        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                        pdf_file.save(file_path)
                        with open(file_path, "rb") as f:
                            pdf_bytes = f.read()
                        source = f"File: {pdf_file.filename}"
                    else:
                        source = f"URL: {pdf_url}"
                        resp = requests.get(pdf_url, timeout=60)
                        resp.raise_for_status()
                        pdf_bytes = resp.content

                    text = extract_text(pdf_bytes)
                    if not text.strip():
                        message = "Could not extract any text from the PDF."
                        error = True
                    else:
                        result = classify(api_key, text)
                        save_whitepaper(
                            result.get("title", "Untitled"),
                            source,
                            result.get("main_category", "Nondescript"),
                            result.get("industry", "Other"),
                            result.get("short_summary", ""),
                            file_path=file_path
                        )
                        return redirect(url_for("index"))
            except Exception as e:
                message = f"Error: {e}"
                error = True

    whitepapers = load_whitepapers()
    if filter_main_category:
        whitepapers = [w for w in whitepapers if w["main_category"] == filter_main_category]
    if filter_industry:
        whitepapers = [w for w in whitepapers if w["industry"] == filter_industry]

    # Sort by created_at string (ISO format sorts correctly lexicographically)
    reverse = (sort_order == "newest")
    whitepapers = sorted(whitepapers, key=lambda w: w["created_at"], reverse=reverse)

    return render_template_string(
        HTML_TEMPLATE,
        message=message,
        error=error,
        whitepapers=whitepapers,
        main_categories=MAIN_CATEGORIES,
        industries=INDUSTRIES,
        filter_main_category=filter_main_category,
        filter_industry=filter_industry,
        sort_order=sort_order,
    )

@app.route("/download/<int:wp_id>")
def download_whitepaper(wp_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM whitepapers WHERE id = ?;", (wp_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        return "No local file available for this whitepaper.", 404
    file_path = row[0]
    if not os.path.isfile(file_path):
        return "File not found on server.", 404
    filename = os.path.basename(file_path)
    return send_file(file_path, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(debug=True)
