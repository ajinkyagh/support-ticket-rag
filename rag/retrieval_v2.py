import os
import sys
import json
import ollama
import anthropic
from supabase import create_client
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Confidence threshold below which we flag low-quality retrieval results
LOW_SIMILARITY_THRESHOLD = 0.6

# System prompt is separated from the user message so Claude has a stable
# persona across turns. Putting rules here (vs. the user message) gives them
# higher weight and prevents the model from "forgetting" them mid-reasoning.
SYSTEM_PROMPT = """You are an expert support ticket analyst with deep knowledge of \
customer service patterns. Your job is to analyze support tickets \
and provide structured, actionable insights.

Rules you must follow:
- Answer ONLY based on the tickets provided in context
- If similarity scores are below 0.6, explicitly flag low confidence
- Always reason step by step before giving your final answer
- Return your response as valid JSON only, no extra text

Response format:
{
  "answer": "your main answer here",
  "confidence": "high / medium / low",
  "relevant_categories": ["list of categories found"],
  "reasoning": "your step by step reasoning here",
  "data_gap": "what data is missing that would improve this answer",
  "suggested_follow_up": "one follow up question the user could ask"
}"""


def get_question() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter your support question: ")


def embed_question(question: str) -> list[float]:
    # Must use the same model that generated the stored embeddings so that
    # the vector space is consistent and cosine similarity is meaningful.
    result = ollama.embeddings(model="nomic-embed-text", prompt=question)
    return result["embedding"]


def search_similar_tickets(embedding: list[float], top_k: int = 5) -> list[dict]:
    # Calls the match_tickets Postgres function which uses pgvector's <=>
    # (cosine distance) operator. Returns similarity = 1 - cosine_distance,
    # so 1.0 is a perfect match and values below 0.6 are considered weak.
    response = supabase.rpc(
        "match_tickets",
        {"query_embedding": embedding, "match_count": top_k}
    ).execute()
    return response.data


def build_user_message(question: str, tickets: list[dict]) -> str:
    # We include similarity scores in the user message (not just the system
    # prompt) so Claude can reason about retrieval quality inline. Flagging
    # low scores here lets the model adjust its confidence output accordingly.
    low_confidence_warning = ""
    if all(t["similarity"] < LOW_SIMILARITY_THRESHOLD for t in tickets):
        low_confidence_warning = (
            "\n⚠️  WARNING: All similarity scores are below 0.6. "
            "The retrieved tickets may not be relevant to this question.\n"
        )

    ticket_lines = []
    for i, t in enumerate(tickets, 1):
        ticket_lines.append(
            f"Ticket {i}:\n"
            f"  Similarity: {t['similarity']:.4f}\n"
            f"  Body: {t['body']}\n"
            f"  Category: {t['category']}\n"
            f"  Priority: {t['priority']}"
        )
    tickets_block = "\n\n".join(ticket_lines)

    # The "think step by step" instruction is placed in the user message
    # (not the system prompt) because it applies to this specific request.
    # Repeating the JSON-only reminder here reduces formatting errors —
    # models sometimes revert to prose when reasoning gets long.
    return (
        f"Question: {question}\n"
        f"{low_confidence_warning}\n"
        f"Retrieved support tickets:\n\n{tickets_block}\n\n"
        "Think step by step before arriving at your final answer. "
        "Consider each ticket's relevance based on its similarity score. "
        "Remember: return valid JSON only, no extra text outside the JSON object."
    )


def ask_claude(question: str, tickets: list[dict]) -> dict:
    user_message = build_user_message(question, tickets)

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()

    # Parse the JSON response. If Claude adds markdown fences despite the
    # instructions, strip them before parsing so we degrade gracefully.
    try:
        # Strip optional ```json ... ``` wrapper
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Return a structured error so the caller can handle it uniformly
        return {
            "answer": "Failed to parse Claude's response as JSON.",
            "confidence": "low",
            "relevant_categories": [],
            "reasoning": raw,
            "data_gap": "N/A",
            "suggested_follow_up": "N/A",
            "_parse_error": str(e),
        }


def print_results(question: str, tickets: list[dict], result: dict) -> None:
    print("\n" + "=" * 60)
    print("TOP 5 SIMILAR TICKETS")
    print("=" * 60)
    for i, t in enumerate(tickets, 1):
        score_flag = " ⚠️  low" if t["similarity"] < LOW_SIMILARITY_THRESHOLD else ""
        print(f"\n[{i}] Similarity: {t['similarity']:.4f}{score_flag}")
        print(f"    Body:     {t['body']}")
        print(f"    Category: {t['category']}")
        print(f"    Priority: {t['priority']}")

    print("\n" + "=" * 60)
    print("CLAUDE'S ANALYSIS")
    print("=" * 60)

    if "_parse_error" in result:
        print(f"\n⚠️  JSON parse error: {result['_parse_error']}")
        print(f"\nRaw response:\n{result['reasoning']}")
        return

    print(f"\nAnswer:              {result.get('answer', 'N/A')}")
    print(f"Confidence:          {result.get('confidence', 'N/A')}")
    print(f"Relevant categories: {', '.join(result.get('relevant_categories', []))}")
    print(f"\nReasoning:\n  {result.get('reasoning', 'N/A')}")
    print(f"\nData gap:            {result.get('data_gap', 'N/A')}")
    print(f"Suggested follow-up: {result.get('suggested_follow_up', 'N/A')}\n")


def main():
    question = get_question()
    print(f"\nSearching for tickets similar to: \"{question}\"")

    embedding = embed_question(question)
    tickets = search_similar_tickets(embedding, top_k=5)

    if not tickets:
        print("No tickets found. Make sure the match_tickets SQL function exists in Supabase.")
        sys.exit(1)

    result = ask_claude(question, tickets)
    print_results(question, tickets, result)


if __name__ == "__main__":
    main()
