"""MCP server for the workshop documents. It runs on Modal.

The server has one tool: search_documents.
The tool finds the chunks that match a query in the Chroma Cloud collection.
The tool gives only the text of the chunks.
"""

import modal

APP_NAME = "workshop-documents-mcp"

# Use the same collection as the "Document Ingestion" notebook.
# The collection uses the Chroma default embedding model (all-MiniLM-L6-v2).
COLLECTION_NAME = "workshop_documents_default"
TOP_K = 3              # Number of chunks that one query gives.
MAX_TOP_K = 10         # Largest number of chunks that one query can give.


def download_model():
    """Download the default embedding model into the image."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    DefaultEmbeddingFunction()(["Download the model."])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("chromadb==1.5.9", "mcp==2.3.0")
    .run_function(download_model)
)

app = modal.App(APP_NAME)

with image.imports():
    import os

    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings


def connect_to_chroma():
    """Connect to Chroma Cloud. Return the client."""
    return chromadb.CloudClient(
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        api_key=os.environ["CHROMA_API_KEY"],
    )


def open_collection():
    """Open the collection that the ingestion notebook made."""
    return connect_to_chroma().get_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )


def retrieve_text(collection, query, top_k=TOP_K):
    """Find the chunks that match the query. Return only their text."""
    result = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents"],
    )
    return result["documents"][0]


def make_server():
    """Make the MCP server and its tool."""
    collection = open_collection()
    server = MCPServer(
        name="Workshop Documents",
        instructions=(
            "Use search_documents to find text in the workshop policy documents. "
            "Give the answer only from the text that the tool gives."
        ),
    )

    @server.tool()
    def search_documents(query: str, top_k: int = TOP_K) -> list[str]:
        """Find the parts of the workshop documents that match the query.

        Args:
            query: The question or the words to find.
            top_k: The number of text chunks to give (1 to 10).
        """
        top_k = max(1, min(top_k, MAX_TOP_K))
        return retrieve_text(collection, query, top_k)

    return server


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv(__file__)],
    scaledown_window=300,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve():
    """Serve the MCP server with the Streamable HTTP transport at the path /mcp."""
    return make_server().streamable_http_app(
        stateless_http=True,
        json_response=True,
        # The Modal URL is not localhost. Thus, turn off the localhost-only host check.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
