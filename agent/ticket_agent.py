import os
import sys
import json
import ollama
import anthropic
from supabase import create_client
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an intelligent support ticket analyst agent.
You have access to three tools:
- search_tickets: for finding specific tickets by meaning
- get_category_trends: for category volume analysis
- get_daily_trends: for time-based volume patterns

Always choose the most appropriate tool based on the question.
After getting tool results, provide a clear, structured answer.
Explain which tool you used and why."""

# ---------------------------------------------------------------------------
# Tool definitions
# Claude's tool use works by sending a `tools` list with each tool's name,
# description, and JSON Schema for its inputs. Claude reads the descriptions
# to decide which tool fits the user's question — so clear descriptions are
# critical for correct tool selection. No code executes on Claude's side;
# we receive a tool_use block and run the function ourselves, then return
# the result in a tool_result block for Claude to compose a final answer.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_tickets",
        "description": (
            "Search support tickets by semantic similarity. "
            "Use this when the user wants to find specific tickets, "
            "examples of issues, or asks about particular problems."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find semantically similar tickets."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_category_trends",
        "description": (
            "Get ticket volume breakdown by category. "
            "Use this when the user asks about which categories have "
            "the most tickets, overall trends, or category distribution."
        ),
        # No inputs required — this tool always returns the full category summary
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_daily_trends",
        "description": (
            "Get ticket volume trends over time. "
            "Use this when the user asks about ticket volume changes, "
            "increases or decreases over time, or time-based patterns."
        ),
        # No inputs required — returns all batch trend rows
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_tickets(query: str) -> list[dict]:
    """Embed the query and retrieve the top 5 semantically similar tickets."""
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    response = supabase.rpc(
        "match_tickets",
        {"query_embedding": embedding, "match_count": 5}
    ).execute()
    return response.data


def get_category_trends() -> list[dict]:
    """Return all rows from the dbt category_summary view."""
    response = supabase.table("category_summary").select("*").execute()
    return response.data


def get_daily_trends() -> list[dict]:
    """Return all rows from the dbt daily_trends view."""
    response = supabase.table("daily_trends").select("*").execute()
    return response.data


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch the tool call Claude requested and return result as a JSON string.

    We serialize to JSON so the tool_result content is a plain string —
    Claude can read structured JSON in text form just as well as native dicts,
    and the Messages API requires tool_result content to be a string or
    content block list.
    """
    if tool_name == "search_tickets":
        result = search_tickets(tool_input["query"])
    elif tool_name == "get_category_trends":
        result = get_category_trends()
    elif tool_name == "get_daily_trends":
        result = get_daily_trends()
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(question: str) -> None:
    print(f"\nQuestion: {question}\n")

    # Step 1 — Send the user's question to Claude along with all tool definitions.
    # Claude will respond with either a text answer (if no tool is needed) or
    # one or more tool_use blocks indicating which tool(s) to call and with
    # what arguments. stop_reason == "tool_use" means Claude wants a tool.
    messages = [{"role": "user", "content": question}]

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )

    # Step 2 — Check if Claude chose a tool. If stop_reason is "end_turn"
    # Claude answered directly without needing a tool.
    if response.stop_reason != "tool_use":
        print("Claude answered directly (no tool needed):\n")
        print(response.content[0].text)
        return

    # Step 3 — Extract the tool_use block from Claude's response.
    # A response can contain a mix of text and tool_use blocks; we find the
    # tool_use block specifically to get the tool name and inputs.
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    tool_name = tool_use_block.name
    tool_input = tool_use_block.input
    tool_use_id = tool_use_block.id  # must be echoed back in tool_result

    print(f"Tool selected : {tool_name}")
    print(f"Tool input    : {json.dumps(tool_input) if tool_input else '(none)'}\n")

    # Step 4 — Execute the tool locally and capture the result.
    tool_result = execute_tool(tool_name, tool_input)

    # Step 5 — Send the tool result back to Claude in a new conversation turn.
    # The message structure requires:
    #   - the assistant turn (Claude's response with the tool_use block)
    #   - a user turn containing a tool_result block with the matching tool_use_id
    # Claude then uses the result to compose its final natural-language answer.
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result
            }
        ]
    })

    # Step 6 — Get Claude's final answer now that it has the tool data.
    final_response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )

    # Extract the text from the final response
    final_text = next(
        (b.text for b in final_response.content if hasattr(b, "text")),
        "No text response returned."
    )

    print("=" * 60)
    print("AGENT ANSWER")
    print("=" * 60)
    print(f"\n{final_text}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter your question: ")
    else:
        question = " ".join(sys.argv[1:])

    run_agent(question)
