"""Telegram bot for the workshop documents. It runs on Modal.

Telegram sends each new message to the web endpoint `webhook`.
The endpoint starts the function `reply` and gives an answer to Telegram at once.
The function `reply` asks Gemini and sends the answer to the chat.

The bot uses the same tools as the "Chat with Docs UI" notebook:
- search_documents: the tool of the MCP server on Modal (the knowledge base).
- web_search: a DuckDuckGo search that the bot calls itself.
"""

import modal

APP_NAME = "workshop-telegram-bot"
MCP_APP_NAME = "workshop-documents-mcp"   # The same name as in mcp_server.py.
STATE_NAME = "workshop-telegram-bot-state"
MODEL = "gemini-flash-lite-latest"
MAX_TOOL_ROUNDS = 5       # Largest number of tool rounds for one question.
MAX_SEARCH_RESULTS = 5    # Number of web results for one search.
MAX_HISTORY = 10          # Number of messages that the bot keeps for each chat.
MAX_MESSAGE_LENGTH = 3500 # Telegram accepts 4096 characters. Keep space for the HTML tags.

DEFAULT_SETTINGS = {"documents": True, "web_search": True}

COMMANDS = [
    {"command": "kb", "description": "Turn the knowledge base on or off"},
    {"command": "web", "description": "Turn the web search on or off"},
    {"command": "settings", "description": "Show the settings"},
    {"command": "reset", "description": "Forget the conversation"},
    {"command": "help", "description": "Show the help"},
]

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "google-genai==2.8.0",
    "mcp==2.3.0",
    "ddgs==9.14.4",
    "httpx==0.28.1",
    "fastapi[standard]==0.135.1",
)

app = modal.App(APP_NAME)

# The settings and the history of each chat. The values stay after a container stops.
state = modal.Dict.from_name(STATE_NAME, create_if_missing=True)

with image.imports():
    import asyncio
    import datetime
    import hashlib
    import hmac
    import html
    import os
    import re

    import httpx
    from ddgs import DDGS
    from fastapi import HTTPException, Request
    from google import genai
    from google.genai import errors, types
    from mcp import Client


SYSTEM_WITH_DOCUMENTS = (
    "You answer questions about the workshop policy documents in a Telegram chat. "
    "Use the search_documents tool to find the text that answers the question. "
    "Give the answer only from the text that the tool gives. "
    "If the text does not contain the answer, say so. "
    "You can search more than one time with different words. "
    "Keep the answer short."
)

SYSTEM_WITHOUT_DOCUMENTS = (
    "You are a helpful assistant in a Telegram chat. Give the answer from your general knowledge. "
    "You do not have access to the workshop documents. "
    "Keep the answer short."
)

SEARCH_INSTRUCTION = (
    " Use the web_search tool when the question needs current or specific facts."
    " The search results are more recent than your knowledge. Trust the search results."
)

HELP_TEXT = (
    "Ask a question about the workshop documents.\n\n"
    "/kb - turn the knowledge base on or off\n"
    "/web - turn the web search on or off (when the knowledge base is off)\n"
    "/settings - show the settings\n"
    "/reset - forget the conversation\n"
    "/help - show this help"
)


# ---------- Telegram ----------

def make_webhook_secret(bot_token):
    """Make the secret that Telegram sends with each update. The secret comes from the bot token."""
    return hashlib.sha256(bot_token.encode()).hexdigest()


async def call_telegram(method, **params):
    """Call a method of the Telegram Bot API. Return the JSON result."""
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/{method}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=params)
    return response.json()


def split_text(text):
    """Divide a long text into parts that Telegram accepts. Divide at the line breaks."""
    parts = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > MAX_MESSAGE_LENGTH:
            parts.append(line[:MAX_MESSAGE_LENGTH])
            line = line[MAX_MESSAGE_LENGTH:]
        if len(current) + len(line) > MAX_MESSAGE_LENGTH:
            parts.append(current)
            current = ""
        current += line
    if current.strip():
        parts.append(current)
    return parts


def to_telegram_html(text):
    """Change the Markdown of Gemini into the HTML that Telegram accepts."""
    text = html.escape(text, quote=False)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*)[*-]\s+", r"\1• ", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(?!\s)([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    return text


async def send_text(chat_id, text):
    """Send a text to the chat. If Telegram does not accept the HTML, send plain text."""
    for part in split_text(text):
        result = await call_telegram("sendMessage", chat_id=chat_id, text=to_telegram_html(part), parse_mode="HTML")
        if not result.get("ok"):
            await call_telegram("sendMessage", chat_id=chat_id, text=part)


# ---------- Chat state ----------

async def load_settings(chat_id):
    """Get the settings of the chat."""
    return await state.get.aio(f"{chat_id}:settings", dict(DEFAULT_SETTINGS))


async def load_history(chat_id):
    """Get the last messages of the chat."""
    return await state.get.aio(f"{chat_id}:history", [])


async def save_history(chat_id, history, question, answer):
    """Add the question and the answer to the history of the chat. Keep only the last messages."""
    history = history + [{"role": "user", "text": question}, {"role": "model", "text": answer}]
    await state.put.aio(f"{chat_id}:history", history[-MAX_HISTORY:])


def describe_settings(settings):
    """Make a text that shows the settings."""
    documents = "on" if settings["documents"] else "off"
    web_search = "on" if settings["web_search"] else "off"
    return f"Knowledge base: {documents}\nWeb search: {web_search} (only when the knowledge base is off)"


# ---------- Tools ----------

WEB_SEARCH_DECLARATION = {
    "name": "web_search",
    "description": "Search the web. Use this tool for current or specific facts.",
    "parameters_json_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search words."}},
        "required": ["query"],
    },
}


async def find_mcp_url():
    """Get the URL of the deployed MCP server."""
    if os.environ.get("MCP_URL"):
        return os.environ["MCP_URL"]
    return await modal.Function.from_name(MCP_APP_NAME, "serve").get_web_url.aio() + "/mcp"


def to_gemini_tool(mcp_tools):
    """Change the MCP tools into one Gemini tool."""
    declarations = [
        types.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters_json_schema=tool.input_schema,
        )
        for tool in mcp_tools
    ]
    return types.Tool(function_declarations=declarations)


async def call_mcp_tool(mcp, function_call):
    """Call the MCP tool that Gemini asks for. Return the list of text chunks and the error flag."""
    result = await mcp.call_tool(function_call.name, dict(function_call.args or {}))
    texts = [content.text for content in result.content if content.type == "text"]
    return texts, result.is_error


async def call_web_search(function_call):
    """Search the web for the query that Gemini gives. Return the list of results and the error flag."""
    query = (function_call.args or {}).get("query", "").strip()
    if not query:
        return ["The query is empty. Give the search words in the query argument."], True
    try:
        # DDGS is not async. Thus, run the search in a thread.
        results = await asyncio.to_thread(DDGS().text, query, max_results=MAX_SEARCH_RESULTS)
    except Exception as error:
        return [f"The search failed: {error}"], True
    texts = [f"{result['title']}\n{result['href']}\n{result['body']}" for result in results]
    return texts, False


def make_config(system_instruction, tool=None):
    """Make the Gemini configuration. The bot calls the tools. Thus, turn off the automatic function calls."""
    # Gemini does not know the date. Without the date, Gemini can think that recent events did not occur.
    system_instruction += f" Today is {datetime.date.today():%d %B %Y}."
    if tool is None:
        return types.GenerateContentConfig(system_instruction=system_instruction)
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def make_tool_note(function_call):
    """Make a line that shows a tool call."""
    query = (function_call.args or {}).get("query", "")
    if function_call.name == "web_search":
        return f"🌐 Searched the web: {query}"
    return f"🔎 Searched the documents: {query}"


# ---------- Gemini ----------

def make_contents(history, question):
    """Make the Gemini contents from the history and the new question."""
    contents = [types.Content(role=item["role"], parts=[types.Part(text=item["text"])]) for item in history]
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    return contents


def get_text(response):
    """Get the answer text from a Gemini response. Do not use the thought parts."""
    if not response.candidates or not response.candidates[0].content:
        return ""
    parts = response.candidates[0].content.parts or []
    return "".join(part.text for part in parts if part.text and not part.thought)


async def answer_from_texts(gemini, contents, system_instruction, texts):
    """Get the answer from the text that the tools gave. Do not give tools to Gemini.

    The contents do not contain the function calls. If the contents contain function calls,
    Gemini can ask for more function calls, also without tools.
    """
    found = "\n\n---\n\n".join(texts) or "The tools gave no text."
    request = (
        f"The tools gave this text:\n\n{found}\n\n"
        "Answer the last question with this text. If the text does not contain the answer, say so."
    )
    contents = contents + [types.Content(role="user", parts=[types.Part(text=request)])]
    system_instruction += " You cannot call tools now. Give the answer as text."
    response = await gemini.aio.models.generate_content(
        model=MODEL, contents=contents, config=make_config(system_instruction),
    )
    return get_text(response)


async def run_tool_loop(gemini, contents, system_instruction, tool, call_tool, notes):
    """Get the answer from Gemini. Call the tools with call_tool when Gemini asks for them."""
    question = list(contents)   # The contents without the function calls.
    config = make_config(system_instruction, tool)
    texts = []
    error_rounds = 0
    for _ in range(MAX_TOOL_ROUNDS):
        response = await gemini.aio.models.generate_content(model=MODEL, contents=contents, config=config)
        calls = response.function_calls or []
        if not calls:
            return get_text(response)
        # Keep all the parts. The parts can have thought signatures that Gemini must get back.
        contents.append(response.candidates[0].content)
        responses = []
        errors_only = True
        for call in calls:
            notes.append(make_tool_note(call))
            call_texts, is_error = await call_tool(call)
            errors_only = errors_only and is_error
            if not is_error:
                texts.extend(call_texts)
            key = "error" if is_error else "result"
            responses.append(types.Part(function_response=types.FunctionResponse(
                id=call.id, name=call.name, response={key: call_texts},
            )))
        contents.append(types.Content(role="user", parts=responses))
        # Gemini can repeat a bad tool call. After two rounds with only errors, stop the tool calls.
        error_rounds += errors_only
        if error_rounds == 2:
            break
    return await answer_from_texts(gemini, question, system_instruction, texts)


async def answer_question(question, history, settings):
    """Get the answer to the question. Return the answer and the list of tool notes."""
    gemini = genai.Client()
    contents = make_contents(history, question)
    notes = []
    if settings["documents"]:
        async with Client(await find_mcp_url()) as mcp:
            tools = await mcp.list_tools()
            answer = await run_tool_loop(
                gemini, contents, SYSTEM_WITH_DOCUMENTS, to_gemini_tool(tools.tools),
                lambda call: call_mcp_tool(mcp, call), notes,
            )
    elif settings["web_search"]:
        tool = types.Tool(function_declarations=[types.FunctionDeclaration(**WEB_SEARCH_DECLARATION)])
        answer = await run_tool_loop(
            gemini, contents, SYSTEM_WITHOUT_DOCUMENTS + SEARCH_INSTRUCTION, tool, call_web_search, notes,
        )
    else:
        response = await gemini.aio.models.generate_content(
            model=MODEL, contents=contents, config=make_config(SYSTEM_WITHOUT_DOCUMENTS),
        )
        answer = get_text(response)
    return answer or "Gemini gave no text.", notes


def first_error(error):
    """Get the first real error from an error group (the MCP client can give error groups)."""
    while getattr(error, "exceptions", None):
        error = error.exceptions[0]
    return error


# ---------- Updates ----------

def is_allowed(message):
    """Find if the sender can use the bot. If TELEGRAM_ALLOWED_USERS is empty, all persons can use it."""
    allowed = os.environ.get("TELEGRAM_ALLOWED_USERS", "").replace(" ", "")
    if not allowed:
        return True
    return str(message.get("from", {}).get("id")) in allowed.split(",")


async def handle_command(chat_id, text):
    """Do the command in the message. Return the reply text."""
    command = text.split()[0].split("@")[0].lower()
    settings = await load_settings(chat_id)
    if command == "/kb":
        settings["documents"] = not settings["documents"]
        await state.put.aio(f"{chat_id}:settings", settings)
        return describe_settings(settings)
    if command == "/web":
        settings["web_search"] = not settings["web_search"]
        await state.put.aio(f"{chat_id}:settings", settings)
        return describe_settings(settings)
    if command == "/settings":
        return describe_settings(settings)
    if command == "/reset":
        await state.put.aio(f"{chat_id}:history", [])
        return "I forgot the conversation."
    return f"{HELP_TEXT}\n\n{describe_settings(settings)}"


async def handle_question(chat_id, question):
    """Answer the question. Return the reply text."""
    await call_telegram("sendChatAction", chat_id=chat_id, action="typing")
    settings = await load_settings(chat_id)
    history = await load_history(chat_id)
    try:
        answer, notes = await answer_question(question, history, settings)
    except errors.APIError as error:
        return f"Gemini error {error.code}: {error.message}"
    except Exception as error:
        error = first_error(error)
        return f"Error: {type(error).__name__}: {error}"
    await save_history(chat_id, history, question, answer)
    if not notes:
        return answer
    return "\n".join(f"_{note}_" for note in notes) + "\n\n" + answer


async def handle_update(update):
    """Reply to one Telegram update. Use only the text messages."""
    message = update.get("message") or {}
    text = message.get("text")
    chat_id = (message.get("chat") or {}).get("id")
    if not text or chat_id is None:
        return
    if not is_allowed(message):
        await send_text(chat_id, "You cannot use this bot.")
        return
    if text.startswith("/"):
        reply_text = await handle_command(chat_id, text)
    else:
        reply_text = await handle_question(chat_id, text)
    await send_text(chat_id, reply_text)


# ---------- Modal functions ----------

@app.function(image=image, secrets=[modal.Secret.from_dotenv(__file__)], timeout=300)
async def reply(update: dict):
    """Reply to one Telegram update."""
    await handle_update(update)


@app.function(image=image, secrets=[modal.Secret.from_dotenv(__file__)])
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="POST")
async def webhook(request: Request):
    """Get an update from Telegram. Start the reply and give an answer to Telegram at once."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not bot_token or not hmac.compare_digest(secret, make_webhook_secret(bot_token)):
        raise HTTPException(status_code=403)
    await reply.spawn.aio(await request.json())
    return {"ok": True}
