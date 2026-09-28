# 📊 AI Data Analyst Agent

> Ask questions about your CSV data in **plain English** — the AI writes the code, your machine runs it. **Your data never leaves your computer.**

Built with Python · Streamlit · Pandas · Ollama (Llama 3.1 8B) · Plotly

---

## 💡 What Does It Do?

1. You **upload a CSV file**
2. The app **auto-cleans** your data (removes duplicates, fixes data types, fills missing values)
3. You **type a question** like *"Which product has the highest revenue?"*
4. A **local AI model** (Ollama) writes pandas code to answer it
5. The code runs **on your machine** — results shown as tables, KPI cards, and charts
6. Optionally **download a PDF** report of the analysis

---

## 🔒 Privacy First

```
Your CSV  →  Auto Cleaning  →  Schema Only  →  Ollama LLM (local)
                                    ↑
              Column names, data types, 3 sample values per column
              ──── Raw data rows are NEVER sent to the AI ────
                                    ↓
                          AI returns Python code
                                    ↓
                     AST Security Check (blocks dangerous ops)
                                    ↓
                     Execute locally on a copy of your data
```

- **No cloud API.** Uses Ollama running locally on your machine.
- **No data leaves.** Only column descriptions go to the model.
- **No blind execution.** Every line of AI code is scanned before running.

---

## 📁 Project Structure

```
AI-Data-Analyst-Agent/
│
├── app.py            ← Streamlit web UI  (start here)
├── agent.py          ← AI brain: schema → LLM → validate → execute
├── data_utils.py     ← CSV loading, auto-cleaning, statistics, PDF export
├── visualization.py  ← KPI cards and automatic Plotly charts
│
├── data/
│   └── sales.csv     ← Built-in sample dataset
│
├── .streamlit/
│   └── config.toml   ← Dark theme + upload size limit
│
├── requirements.txt
└── README.md
```

**Only 4 Python files.** Each one has a single clear job.

---

## ⚙️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| Streamlit | 1.50+ | Web UI |
| Pandas | 2.2+ | Data manipulation |
| Ollama | 0.4+ | Local LLM runtime |
| Llama 3.1 8B | — | The AI model |
| Plotly | 5.20+ | Interactive charts |
| ReportLab | 4.2+ | PDF generation |

---

## 🚀 Setup & Run

### Step 1 — Install Ollama

Download from [ollama.ai](https://ollama.ai) and install it.  
Then pull the model:

```bash
ollama pull llama3.1:8b
```

### Step 2 — Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/AI-Data-Analyst-Agent.git
cd AI-Data-Analyst-Agent
```

### Step 3 — Create virtual environment & install packages

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 💬 Example Questions to Ask

Once you load a dataset, try asking:

- *"What is the total revenue by product?"*
- *"Show me the top 5 salespersons by profit."*
- *"What is the average discount given per region?"*
- *"Which month had the highest sales?"*
- *"Is there a correlation between quantity and revenue?"*

---

## 🛡️ Security Design

The AI-generated code goes through a **3-layer security check** before execution:

1. **AST Parsing** — code is parsed into a syntax tree and every node is inspected
2. **Blocklist check** — any reference to `os`, `sys`, `open`, `eval`, `exec`, `subprocess`, `requests`, etc. is rejected
3. **Sandboxed execution** — code runs in an isolated namespace with only pandas and safe builtins

---

## 🔧 Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Ollama model | `agent.py` → `MODEL` | `llama3.1:8b` |
| Ollama host | `agent.py` → `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| Max upload size | `.streamlit/config.toml` | 50 MB |
| Max rows | `data_utils.py` → `MAX_ROWS` | 250,000 |

---

## 📄 License

MIT License — free to use, modify, and share.
