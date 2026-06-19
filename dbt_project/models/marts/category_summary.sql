-- Mart model summarising ticket volume and intent diversity per category.
-- Useful for understanding which support categories are most active and
-- how many distinct customer intents each category covers.

select
    category,
    count(*)                    as ticket_count,
    count(distinct intent)      as distinct_intent_count
from {{ ref('stg_tickets') }}
group by category
order by ticket_count desc
