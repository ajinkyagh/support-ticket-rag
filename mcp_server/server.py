"""
MCP (Model Context Protocol) server for the Support Ticket RAG system.

MCP is an open protocol that lets AI assistants (like Claude) call external
tools defined by a server. The server exposes tools with typed inputs; the
client (Claude Code or the Claude desktop app) discovers them automatically
and calls them during a conversation when they're relevant.

This server uses stdio transport — the client launches this script as a
subprocess and communicates over stdin/stdout. No HTTP port is needed.

Tools exposed:
  - search_tickets       : semantic RAG search via pgvector
  - get_category_trends  : category volume from dbt view
  - get_daily_trends     : time-based volume from dbt view
"""

import os
import json
import ollama
from supabase import create_client
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Bootstrap — credentials and clients
# ---------------------------------------------------------------------------

# Load .env from the project root (one level up from this file)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ---------------------------------------------------------------------------
# Create the MCP server
# The name appears in the Claude Code tool picker and in error messages.
# ---------------------------------------------------------------------------

mcp = FastMCP("support-ticket-rag")

# ---------------------------------------------------------------------------
# Tool 1 — search_tickets
# Embeds the query with the same Ollama model used during ingestion so the
# vector space is consistent, then calls the match_tickets Postgres function
# which uses pgvector's <=> cosine distance operator to rank results.
# ---------------------------------------------------------------------------

@mcp.tool()
def search_tickets(query: str) -> str:
    """Search support tickets by semantic similarity using RAG.
    Use when finding specific ticket examples or issues by meaning."""

    # Generate a 768-dim embedding for the user's query
    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    # Retrieve top 5 tickets by cosine similarity via the Supabase RPC function
    response = supabase.rpc(
        "match_tickets",
        {"query_embedding": embedding, "match_count": 5}
    ).execute()

    tickets = response.data
    if not tickets:
        return "No similar tickets found."

    # Format results as readable text for Claude to reason over
    lines = [f"Top {len(tickets)} tickets similar to: '{query}'\n"]
    for i, t in enumerate(tickets, 1):
        lines.append(
            f"[{i}] Similarity: {t['similarity']:.4f}\n"
            f"    Category : {t['category']}\n"
            f"    Priority : {t['priority']}\n"
            f"    Body     : {t['body']}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — get_category_trends
# Queries the dbt category_summary view (materialized in Supabase public
# schema by `dbt run`). Returns ticket counts and distinct intent counts
# per category so Claude can reason about distribution.
# ---------------------------------------------------------------------------

@mcp.tool()
def get_category_trends() -> str:
    """Get ticket volume by category from dbt models.
    Use when asked about category distribution or which category has most tickets."""

    response = supabase.table("category_summary").select("*").execute()
    rows = response.data

    if not rows:
        return "No category data found. Ensure dbt models have been run."

    lines = ["Ticket volume by category:\n"]
    for row in rows:
        lines.append(
            f"  {row['category']:<12} "
            f"tickets: {row['ticket_count']:>4}   "
            f"distinct intents: {row['distinct_intent_count']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3 — get_daily_trends
# Queries the dbt daily_trends view which groups tickets into sequential
# batches of 50 (simulating time periods since there's no created_at column)
# and computes a cumulative count so volume trajectory is visible.
# ---------------------------------------------------------------------------

@mcp.tool()
def get_daily_trends() -> str:
    """Get ticket volume trends over time from dbt models.
    Use when asked about volume changes over time."""

    response = supabase.table("daily_trends").select("*").execute()
    rows = response.data

    if not rows:
        return "No trend data found. Ensure dbt models have been run."

    lines = ["Ticket volume trends (batches of 50 tickets = 1 period):\n"]
    lines.append(f"  {'Batch':<8} {'In Batch':>10} {'Cumulative':>12}")
    lines.append("  " + "-" * 32)
    for row in rows:
        lines.append(
            f"  {row['batch_number']:<8} "
            f"{row['tickets_in_batch']:>10}   "
            f"{row['cumulative_ticket_count']:>10}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point — run the server on stdio transport
# stdio is the standard transport for Claude Code MCP servers: Claude Code
# launches this script as a subprocess and communicates over stdin/stdout.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
