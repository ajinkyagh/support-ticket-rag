-- Mart model simulating ticket volume trends over time.
-- Because the tickets table has no created_at timestamp, we use the
-- auto-incremented id to group rows into sequential batches of 50,
-- treating each batch as a logical "period" (e.g. an ingestion run).
-- The cumulative count shows total tickets processed up to each batch.

with batched as (
    select
        -- Integer division buckets every 50 consecutive ids into one group
        floor((id - 1) / 50) + 1    as batch_number,
        id
    from {{ ref('stg_tickets') }}
),

batch_counts as (
    select
        batch_number,
        count(*) as tickets_in_batch
    from batched
    group by batch_number
)

select
    batch_number,
    tickets_in_batch,
    sum(tickets_in_batch) over (
        order by batch_number
        rows between unbounded preceding and current row
    ) as cumulative_ticket_count
from batch_counts
order by batch_number
