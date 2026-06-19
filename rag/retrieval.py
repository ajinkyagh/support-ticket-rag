import os
import sys
import ollama
import anthropic
from supabase import create_client
from dotenv import load_dotenv

# Step 1: Load credentials from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Step 2: Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_question():
    # Accept question as a command-line argument or prompt interactively
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter your support question: ")

def embed_question(question: str) -> list[float]:
    # Convert the user's question into a 768-dim vector using the same
    # model used to embed the tickets, so the similarity search is meaningful
    result = ollama.embeddings(model="nomic-embed-text", prompt=question)
    return result["embedding"]

def search_similar_tickets(embedding: list[float], top_k: int = 5) -> list[dict]:
    # Use pgvector's cosine distance operator (<=>) via a Postgres RPC function.
    # Cosine distance ranges from 0 (identical) to 2 (opposite); lower is better.
    # We call a Supabase SQL function `match_tickets` defined as:
    #
    #   create or replace function match_tickets(
    #     query_embedding vector(768),
    #     match_count int
    #   )
    #   returns table (
    #     id bigint, ticket_id text, subject text, body text,
    #     category text, priority text,
    #     similarity float
    #   )
    #   language sql stable as $$
    #     select id, ticket_id, subject, body, category, priority,
    #            1 - (embedding <=> query_embedding) as similarity
    #     from tickets
    #     order by embedding <=> query_embedding
    #     limit match_count;
    #   $$;
    #
    # 1 - cosine_distance gives a similarity score where 1.0 = perfect match.
    response = supabase.rpc(
        "match_tickets",
        {"query_embedding": embedding, "match_count": top_k}
    ).execute()
    return response.data

def format_tickets_as_context(tickets: list[dict]) -> str:
    # Build a readable context block to pass to Claude
    lines = []
    for i, t in enumerate(tickets, 1):
        lines.append(
            f"Ticket {i}:\n"
            f"  Body: {t['body']}\n"
            f"  Category: {t['category']}\n"
            f"  Priority: {t['priority']}\n"
            f"  Similarity: {t['similarity']:.4f}"
        )
    return "\n\n".join(lines)

def ask_claude(question: str, context: str) -> str:
    # Step 5: Send the retrieved tickets as context to Claude and get an answer
    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a support ticket analyst. Answer the user's question "
            "based only on the support tickets provided as context. "
            "Be specific and reference the tickets directly."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Context (retrieved support tickets):\n\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return message.content[0].text

def main():
    # Step 3: Get the user's question
    question = get_question()
    print(f"\nSearching for tickets similar to: \"{question}\"\n")

    # Step 4: Embed the question
    embedding = embed_question(question)

    # Step 5: Retrieve top 5 similar tickets from Supabase via pgvector
    tickets = search_similar_tickets(embedding, top_k=5)

    if not tickets:
        print("No tickets found. Make sure the match_tickets SQL function is created in Supabase.")
        sys.exit(1)

    # Step 6: Print the retrieved tickets
    print("=" * 60)
    print("TOP 5 SIMILAR TICKETS")
    print("=" * 60)
    for i, t in enumerate(tickets, 1):
        print(f"\n[{i}] Similarity: {t['similarity']:.4f}")
        print(f"    Body:     {t['body']}")
        print(f"    Category: {t['category']}")
        print(f"    Priority: {t['priority']}")

    # Step 7: Build context and ask Claude
    context = format_tickets_as_context(tickets)
    print("\n" + "=" * 60)
    print("CLAUDE'S ANSWER")
    print("=" * 60)
    answer = ask_claude(question, context)
    print(f"\n{answer}\n")

if __name__ == "__main__":
    main()
