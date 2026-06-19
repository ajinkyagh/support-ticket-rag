-- Staging model for the raw tickets table.
-- Selects all columns except the embedding vector (too large for analytics use),
-- renames priority to intent to reflect the source data's original meaning,
-- and filters out any rows with a null body to ensure downstream models
-- only process complete records.

select
    id,
    ticket_id,
    subject,
    body,
    category,
    priority as intent
from {{ source('public', 'tickets') }}
where body is not null
