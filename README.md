# Workshop 2 — Chat with Your Documents (RAG + MCP + Modal)

Build a small Retrieval-Augmented Generation (RAG) system for a set of policy PDFs, and use it from three places:
Claude (as an MCP connector), a Gradio web chat, and a Telegram bot.

```
                 ┌──────────────────────┐
  PDFs ──[1]──▶  │  Chroma Cloud         │  ◀── search_documents ──┐
  (OCR, chunk,   │  collection           │                         │
   embed)        └──────────────────────┘                         │
                                                          ┌────────┴─────────┐
                                                          │  MCP server [2]  │  (Modal)
                                                          └────────┬─────────┘
                                   ┌───────────────────────────────┼───────────────────────┐
                                   ▼                               ▼                       ▼
                         Claude Code / Claude.ai        Gradio chat UI [3]        Telegram bot [4] (Modal)
                                                        (Gemini + web search)     (Gemini + web search)
```

## What is inside

| Notebook | What it does |
|---|---|
| `[1] Document Ingestion.ipynb` | Reads the PDFs in `documents/` (with OCR for scanned pages), divides the text into chunks, and writes them to a Chroma Cloud collection. Ends with a retrieval test. |
| `[2] MCP_Server_Modal.ipynb` | Writes `mcp_server.py`, an MCP server with one tool, `search_documents`. Tests it locally, deploys it to Modal, and shows how to connect it to Claude. |
| `[3] Chat with Docs UI.ipynb` | A Gradio chat app on Gemini (`gemini-flash-lite-latest`). It has a **Use the knowledge base** toggle (MCP tool) and a **Use web search** toggle (DuckDuckGo). |
| `[4] Telegram Bot.ipynb` | Writes `telegram_bot.py`, a Telegram bot with the same features as the chat app. It runs on Modal through a webhook. Commands: `/kb`, `/web`, `/settings`, `/reset`, `/help`. |

The notebooks generate `mcp_server.py` and `telegram_bot.py` with `%%writefile`. If you change a notebook cell, run it again to update the file.

The `documents/` folder has the sample documents:

- Cambodia Financial Technology Development Policy 2023–2028
- MEF IT Architecture Framework
- E-Services for Business (ESB) Strategy, March 2025

## Prerequisites

- **Python 3.10 or later** and Jupyter (VS Code, JupyterLab or Notebook)
- Free accounts on these services:
  - [Chroma Cloud](https://trychroma.com): the vector database
  - [Modal](https://modal.com): hosts the MCP server and the bot
  - [Google AI Studio](https://aistudio.google.com/apikey): Gemini API key, for notebooks [3] and [4]
  - Telegram: for notebook [4] only
- **Optional:** [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR of scanned PDFs (`brew install tesseract` on macOS, `sudo apt install tesseract-ocr` on Ubuntu)

## Step-by-step setup

### Step 1 — Get the code

```bash
git clone <this-repo-url>
cd "Workshop 2"
```

### Step 2 — Make a virtual environment and install the packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install ipykernel ipywidgets   # To run the notebooks in this environment
```

In Jupyter or VS Code, select the `.venv` kernel. Each notebook also has a `%pip install` cell, so you can skip that cell if you used `requirements.txt`.

### Step 3 — Make the `.env` file

```bash
cp .env.example .env
```

Fill in your values. **Do not commit `.env`.** `.gitignore` already excludes it.

| Variable | Where to get it | Used by |
|---|---|---|
| `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` | Chroma Cloud dashboard → make a database → **Connect** | [1], [2] |
| `GEMINI_API_KEY` | Google AI Studio → **Get API key** | [3], [4] |
| `TELEGRAM_BOT_TOKEN` | Telegram **@BotFather** → `/newbot` | [4] |
| `TELEGRAM_ALLOWED_USERS` | Optional. Comma-separated user IDs (get yours from **@userinfobot**). Empty means everyone can use the bot. | [4] |
| `MCP_URL` | Optional. If you leave it empty, the code gets the URL from the deployed Modal app. | [3], [4] |

The Modal deployments copy `.env` into a Modal secret (`modal.Secret.from_dotenv`). After you change `.env`, deploy again.

### Step 4 — Log in to Modal (one time)

```bash
python -m modal setup
```

This opens a browser window. You can also run it from the commented cell in notebook [2].

### Step 5 — Ingest the documents: notebook [1]

Run all cells. The notebook:

1. Connects to Chroma Cloud and makes the collection `workshop_documents_default`. The first run downloads the default embedding model (all-MiniLM-L6-v2).
2. Reads each PDF. `OCR_MODE = "auto"` uses OCR only on pages with almost no text. For Khmer documents, install the Khmer Tesseract data and set `OCR_LANGUAGE = "eng+khm"`.
3. Writes the chunks (1000 characters, 200 overlap) with `upsert`. You can run it again without duplicates.
4. Runs three test queries.

To use your own documents, put PDFs into `documents/` and run the notebook again.

### Step 6 — Deploy the MCP server: notebook [2]

Run all cells. The notebook writes `mcp_server.py`, tests it in-process, then runs:

```bash
python -m modal deploy mcp_server.py
```

It prints the endpoint, for example `https://<workspace>--workshop-documents-mcp-serve.modal.run/mcp`.

**Connect it to Claude:**

- **Claude Code:** `claude mcp add --transport http workshop-documents <MCP URL>`
- **Claude.ai / Claude Desktop:** *Settings → Connectors → Add custom connector*, then paste the MCP URL.

### Step 7 — Run the chat UI: notebook [3]

Run all cells. The app opens in the notebook and at <http://127.0.0.1:7860>. The toggles are under **Additional Inputs**:

- **Knowledge base on:** Gemini answers only from the documents (through the MCP tool).
- **Knowledge base off:** Gemini answers from general knowledge. It can also use web search if that toggle is on.

Set `share=True` in `demo.launch()` to get a public link. Run the last cell (`demo.close()`) to stop.

### Step 8 — Deploy the Telegram bot: notebook [4]

1. Make the bot with **@BotFather** and put the token in `.env` (see step 3).
2. Run the cells. They check the token, write `telegram_bot.py`, test the bot locally, and deploy it:
   ```bash
   python -m modal deploy telegram_bot.py
   ```
3. The **Connect Telegram** cell registers the webhook (with a secret token) and the bot commands.
4. Open your bot in Telegram, push **Start**, and ask a question.

Debugging: `modal app logs workshop-telegram-bot`, or run the `getWebhookInfo` cell.

## Stop and clean up

```bash
modal app stop workshop-documents-mcp     # MCP server
modal app stop workshop-telegram-bot      # Telegram bot (run the notebook's last cell first to delete the webhook)
```

## Security notes

- **The MCP endpoint has no authentication.** Anyone with the URL can search the collection. Use only public documents, and stop the app when you do not need it.
- If `TELEGRAM_ALLOWED_USERS` is empty, everyone can use the bot, and they all use your Gemini quota.
- The Telegram webhook rejects requests that do not have the secret token (derived from the bot token).
- Keep `.env` private. Before you push, clear the notebook outputs, because they can contain your Modal URLs and Chroma tenant ID:
  ```bash
  jupyter nbconvert --clear-output --inplace *.ipynb
  ```

## Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError: 'CHROMA_TENANT'` (or a similar key) | The `.env` file is missing or not in the notebook's working folder. |
| `Collection ... does not exist` in [2] | Run notebook [1] first. |
| Scanned pages give no text | Install Tesseract, or set `OCR_MODE`. |
| The first answer is slow | Modal is starting a cold container. Later requests are faster. |
| Modal function not found in [3]/[4] | Deploy `mcp_server.py` (notebook [2]) first, or set `MCP_URL` in `.env`. |
| The bot does not reply | Check `modal app logs workshop-telegram-bot` and the `getWebhookInfo` cell. Deploy again after you change `.env`. |
| `IProgress not found` warning in [3] | `pip install ipywidgets` |

## Project structure

```
.
├── [1] Document Ingestion.ipynb
├── [2] MCP_Server_Modal.ipynb
├── [3] Chat with Docs UI.ipynb
├── [4] Telegram Bot.ipynb
├── mcp_server.py          # Generated by notebook [2]
├── telegram_bot.py        # Generated by notebook [4]
├── documents/             # Source PDFs
├── requirements.txt
├── .env.example           # Copy to .env
└── .gitignore
```
