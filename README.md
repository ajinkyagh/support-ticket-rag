# Support Ticket RAG System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Supabase](https://img.shields.io/badge/Supabase-pgvector-green?logo=supabase)
![dbt](https://img.shields.io/badge/dbt-1.12-orange?logo=dbt)
![Ollama](https://img.shields.io/badge/Ollama-nomic--embed--text-black?logo=ollama)
![Claude](https://img.shields.io/badge/Claude-claude--sonnet--4--6-purple?logo=anthropic)

---

## Overview

Customer support teams receive thousands of tickets but struggle to extract patterns, find similar past issues, or get instant answers from their ticket history. This project builds a full RAG (Retrieval-Augmented Generation) pipeline on top of 900 support tickets across 5 categories — turning a flat CSV into a searchable, queryable knowledge base that an AI agent can reason over.

It does this by embedding each ticket into a 768-dimensional vector using Ollama, storing vectors in Supabase with pgvector, running analytical dbt models on top, and exposing everything through a Claude-powered agent that picks the right tool for each question.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────┐
│            Claude Agent (claude-sonnet-4-6)  │
│     Reads question → selects best tool       │
└──────────┬──────────────┬───────────────────┘
           │              │              │
           ▼              ▼              ▼
  search_tickets   get_category_    get_daily_
                     trends          trends
           │              │              │
           ▼              ▼              ▼
  Ollama embed      Supabase          Supabase
  → pgvector        category_summary  daily_trends
    cosine search   (dbt view)        (dbt view)
           │              │              │
           └──────────────┴──────────────┘
                          │
                          ▼
               Structured JSON Answer
               (confidence + reasoning
                + data gap + follow-up)
```

---

## Tech Stack

| Tool | Purpose | Why this choice |
|---|---|---|
| **Python 3.11** | Pipeline orchestration and scripting | Standard for ML/data workflows; rich ecosystem |
| **Supabase** | Postgres database + REST API | Managed Postgres with pgvector built-in; no infra to run |
| **pgvector** | Vector similarity search (`<=>` cosine distance) | Native Postgres extension; no separate vector DB needed |
| **Ollama** | Local embedding generation | Free, private, no API limits; `nomic-embed-text` gives 768-dim vectors |
| **nomic-embed-text** | Text → vector embeddings | Strong performance, small footprint, 768 dims matches schema |
| **dbt** | SQL-based data transformation and testing | Adds versioned, tested analytics models on top of raw tickets |
| **Claude API** | LLM reasoning, tool use, and JSON generation | Best-in-class tool use and structured output reliability |
| **MCP (Model Context Protocol)** | Expose RAG tools to Claude Code | Standard protocol for connecting AI assistants to external tools |
| **FastMCP** | MCP server framework | Clean Python decorator API; handles stdio transport boilerplate |

---

## Features

- **Semantic Search via RAG** — questions are embedded and matched against 900 tickets using cosine similarity; not keyword search
- **Three RAG patterns demonstrated** — vector similarity (`search_tickets`), aggregation retrieval (`get_category_trends`), and time-series retrieval (`get_daily_trends`)
- **Structured JSON output with confidence scoring** — Claude returns a typed response with `answer`, `confidence`, `reasoning`, `data_gap`, and `suggested_follow_up` fields
- **Chain-of-thought reasoning** — the agent is explicitly prompted to reason step by step before committing to an answer, improving accuracy on ambiguous queries
- **Agentic tool selection** — Claude reads the question and decides which of the 3 tools to call; no hardcoded routing
- **dbt data models with 7 schema tests** — `stg_tickets`, `category_summary`, and `daily_trends` with `unique`, `not_null`, and `accepted_values` tests
- **MCP server integration** — all three tools are exposed via a FastMCP stdio server and registered in `.mcp.json` for Claude Code
- **Data augmentation** — 400 synthetic tickets generated across 4 underrepresented categories to address training imbalance, using 25 realistic templates per category with phrasing variations

---

## Project Structure

```
support-ticket-rag/
│
├── data/
│   ├── tickets.csv                  # 500 original ORDER tickets from Hugging Face
│   ├── tickets_augmented.csv        # 900 tickets across 5 categories (gitignored)
│   ├── download_dataset.py          # Downloads and saves Hugging Face dataset
│   └── augment_dataset.py           # Generates 400 synthetic tickets (100/category)
│
├── embeddings/
│   └── embed_tickets.py             # Embeds each ticket via Ollama, inserts to Supabase
│
├── rag/
│   ├── retrieval.py                 # Basic RAG: vector search + Claude answer
│   └── retrieval_v2.py              # Improved RAG: chain-of-thought + structured JSON
│
├── agent/
│   └── ticket_agent.py              # Agentic loop: Claude picks tool, executes, answers
│
├── mcp_server/
│   └── server.py                    # FastMCP server exposing 3 tools over stdio
│
├── dbt_project/
│   ├── dbt_project.yml              # dbt project config and materialization settings
│   ├── profiles.yml                 # Supabase connection via pooler + env vars
│   └── models/
│       ├── staging/
│       │   ├── stg_tickets.sql      # Cleans raw tickets, drops embedding column
│       │   ├── stg_tickets.yml      # Schema tests: unique, not_null, accepted_values
│       │   └── sources.yml          # Declares public.tickets as dbt source
│       └── marts/
│           ├── category_summary.sql # Ticket count + distinct intents per category
│           ├── daily_trends.sql     # Batch-based volume trend with cumulative count
│           └── marts.yml            # Schema tests for category_summary
│
├── schema.sql                       # Supabase table + pgvector extension setup
├── .mcp.json                        # Claude Code MCP server registration
├── .env                             # Credentials (gitignored)
├── .gitignore
└── requirements.txt
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed locally
- A [Supabase](https://supabase.com) project with pgvector enabled
- An Anthropic API key

### 1. Clone the repository
```bash
git clone https://github.com/ajinkyagh/support-ticket-rag.git
cd support-ticket-rag
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Ollama
```bash
ollama pull nomic-embed-text
ollama serve   # runs on localhost:11434 by default
```

### 5. Set up Supabase
Run the contents of `schema.sql` in your Supabase SQL editor to create the `tickets` table and enable pgvector.

Then create the vector similarity function:
```sql
create or replace function match_tickets(
  query_embedding vector(768),
  match_count int
)
returns table (
  id bigint, ticket_id text, subject text, body text,
  category text, priority text, similarity float
)
language sql stable as $$
  select id, ticket_id, subject, body, category, priority,
         1 - (embedding <=> query_embedding) as similarity
  from tickets
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

### 6. Configure environment variables
Copy `.env` and fill in your credentials:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_DB_PASSWORD=your_db_password
SUPABASE_HOST=db.your-project.supabase.co
SUPABASE_POOLER_HOST=aws-0-us-east-1.pooler.supabase.com
SUPABASE_POOLER_USER=postgres.your-project-ref
ANTHROPIC_API_KEY=sk-ant-...
```

### 7. Download and augment the dataset
```bash
python data/download_dataset.py        # saves data/tickets.csv (500 rows)
python data/augment_dataset.py         # saves data/tickets_augmented.csv (900 rows)
```

### 8. Embed tickets into Supabase
```bash
python embeddings/embed_tickets.py
# prints: Embedded 50/900 ... Embedded 900/900
```

### 9. Run dbt models
```bash
cd dbt_project
dbt run --profiles-dir .       # creates stg_tickets, category_summary, daily_trends
dbt test --profiles-dir .      # runs 7 schema tests
cd ..
```

### 10. Run the agent
```bash
python agent/ticket_agent.py "Which support category has the most tickets?"
python agent/ticket_agent.py "Show me examples of login problems"
python agent/ticket_agent.py "How has ticket volume changed over time?"
```

### 11. Connect the MCP server to Claude Code
The `.mcp.json` file is already configured. Open this project directory in Claude Code and approve the `support-ticket-rag` server when prompted. All three tools will be available in your Claude Code session automatically.

---

## Example Outputs

### High Confidence — Semantic Search
```
Question: "How do customers ask about cancelling an order?"

[1] Similarity: 0.8731
    Body:     I need help cancelling my purchase. The order number is...
    Category: ORDER
    Priority: cancel_order

Answer: {
  "answer": "Customers typically ask about cancellation by referencing their
             order number and expressing urgency. Common patterns include
             direct requests ('I need to cancel'), status checks ('has my
             cancellation gone through'), and procedural questions ('how do
             I cancel an order I just placed').",
  "confidence": "high",
  "relevant_categories": ["ORDER"],
  "reasoning": "Tickets 1-4 all have similarity above 0.85 and directly
                address cancellation intent. The language patterns are
                consistent across the retrieved examples.",
  "data_gap": "No timestamp data to determine if cancellation requests
               spike after promotions or holiday periods.",
  "suggested_follow_up": "What is the typical resolution time for
                          cancellation requests?"
}
```

### Low Confidence — Data Gap Flagged
```
Question: "What payment processor errors do customers report most?"

[1] Similarity: 0.5821
    Body:     My payment keeps failing even though my card is correct.
    Category: PAYMENT
    Priority: payment_failed

Answer: {
  "answer": "Based on the retrieved tickets, card declines and double
             charges are the most commonly reported payment issues.
             However, similarity scores are below 0.6 — these may not
             be the most representative examples.",
  "confidence": "low",
  "relevant_categories": ["PAYMENT"],
  "reasoning": "The similarity scores (0.58, 0.55, 0.54) are all below
                the 0.6 threshold. The tickets discuss payment failures
                generally but do not mention specific processor error
                codes or gateway names.",
  "data_gap": "Tickets lack payment processor names, error codes, or
               gateway identifiers. A dataset with structured payment
               metadata would dramatically improve this answer.",
  "suggested_follow_up": "Which payment methods do customers report
                          having the most trouble with?"
}
```

### Agent Tool Selection — Category Trends
```
Question: "Which support category has the most tickets?"

Tool selected : get_category_trends
Tool input    : (none)

Answer:
Based on the category_summary dbt model, ORDER dominates with 500 tickets
(56% of all tickets), followed by ACCOUNT, PAYMENT, SHIPPING, and REFUND
at 100 tickets each. The ORDER category also has the most distinct intents
(7), suggesting it covers a wider variety of customer problems than the
other categories.
```

---

## Prompt Engineering

### System prompt design
The system prompt is kept in a separate `SYSTEM_PROMPT` constant rather than embedded in the user message. This gives it persistent authority across the conversation — models treat system-level instructions as higher-weight constraints than user-turn instructions.

### Why structured JSON output
Plain prose answers are hard to parse programmatically. By requiring a specific JSON schema (`answer`, `confidence`, `reasoning`, `data_gap`, `suggested_follow_up`), downstream code can read confidence scores, route low-confidence results to human review, and surface data gaps to a data engineering backlog — all without parsing freeform text.

### Chain-of-thought improves accuracy
The instruction "think step by step before answering" is placed in the user message (not the system prompt) because it applies to this specific request, not all requests. Forcing explicit reasoning before the final answer reduces hallucination on ambiguous queries — the model commits its reasoning to text before it commits to a conclusion.

### Confidence scoring logic
The threshold of 0.6 cosine similarity was chosen empirically. Scores above 0.75 reliably indicate the retrieved tickets are genuinely similar to the question. Scores between 0.6 and 0.75 are usable but uncertain. Below 0.6, the retrieved tickets are likely from a different domain and the answer should be treated as speculative. This threshold is surfaced to Claude in the user message so it can calibrate its own confidence output accordingly.

---

## Key Concepts Demonstrated

- **RAG architecture** — the pattern of retrieve-then-generate: fetch relevant context from a database, then have the LLM answer using only that context, reducing hallucination
- **Vector embeddings and similarity search** — text is projected into a high-dimensional space where semantic meaning maps to geometric proximity; pgvector's `<=>` operator finds the nearest vectors efficiently using an HNSW index
- **Tokenization and context window considerations** — the embedding model (`nomic-embed-text`) has a 8192-token context window; ticket bodies are kept under 512 tokens to ensure full semantic capture without truncation
- **Agentic frameworks and tool use** — Claude's tool use API lets the model choose when and which tools to call based on tool descriptions alone; this is the foundation of autonomous agents
- **Data augmentation with LLMs** — when real labeled data is scarce, synthetic tickets generated from curated templates can correct class imbalance; the 25-template-per-category approach with phrasing variation avoids exact duplicates
- **dbt data modeling and testing** — `stg_tickets` separates raw ingestion from analytics concerns; `category_summary` and `daily_trends` are mart-layer views with 7 schema tests ensuring data quality before any dashboard or downstream query consumes them

---

## Known Limitations

- **Synthetic data has structural duplicates** — the 400 augmented tickets cycle through 25 templates with 5 phrasing variations, so some tickets are close paraphrases of each other; this inflates similarity scores within those categories
- **Dataset imbalance persists** — ORDER still has 500 tickets vs. 100 for each other category; a balanced dataset would require either downsampling ORDER or generating 400 more tickets per other category
- **Local Ollama means no cloud deployment** — the embedding step and MCP server both require Ollama running locally; deploying this as a cloud API would require switching to a hosted embedding model (e.g., Voyage AI or OpenAI `text-embedding-3-small`)
- **No authentication on MCP server** — the FastMCP stdio server has no auth layer; it should not be exposed as an HTTP service without adding API key validation

---

## Future Improvements

- **Deploy MCP server to cloud** — wrap `server.py` in a FastAPI HTTP layer with API key auth and deploy to Railway or Fly.io so Claude Code can connect remotely without a local Ollama instance
- **Add hybrid search** — combine pgvector cosine similarity with Postgres full-text search (`tsvector`) using Reciprocal Rank Fusion; keyword matches for exact order numbers or error codes, semantic matches for intent
- **Real ticket dataset with timestamps** — replace the simulated `batch_number` trend proxy with actual `created_at` timestamps to enable real time-series analysis and seasonality detection
- **Add memory to the agent for multi-turn conversations** — currently each `run_agent()` call is stateless; adding a conversation history buffer would let users ask follow-up questions like "show me 5 more examples" or "now filter by PAYMENT only"
