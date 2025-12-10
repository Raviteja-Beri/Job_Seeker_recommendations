import json
import pdfplumber
import docx
import requests


def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    elif "document" in uploaded_file.type:
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])

    return uploaded_file.read().decode("utf-8", errors="ignore")


def clean_json_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def json_or_none(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return None


def extract_json_array(text: str):
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    return json_or_none(match.group(0))


def ollama_available(timeout=1.0):
    try:
        r = requests.get("http://localhost:11434/api/ping", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def parse_json(text: str):
    """Try parse text into JSON list of roles. Returns list or None.

    Steps:
    - clean fenced code blocks
    - try full JSON parse (handles lists or dicts)
    - if dict, look for first list value
    - fall back to extracting the first JSON array substring
    """
    t = clean_json_fences(text)
    parsed = json_or_none(t)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return extract_json_array(t)
