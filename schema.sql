-- Enable pgvector extension
create extension if not exists vector;

-- Create tickets table
create table tickets (
  id bigserial primary key,
  ticket_id text,
  subject text,
  body text,
  category text,
  priority text,
  embedding vector(768)
);
