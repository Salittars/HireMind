-- Staging: one clean row per real posting.
-- 1) keep one row per (source, posting_id)
-- 2) drop cross-posts: same title + same company counts once (Arbeitnow lists one job per city).
--    Names are compared lower-cased with punctuation/spaces removed ("Mirakl - Labs" = "mirakllabs").
with by_id as (
    select *,
        row_number() over (
            partition by source, posting_id
            order by published_at desc nulls last
        ) as rn
    from raw_postings
),

by_title_company as (
    select * exclude (rn),
        row_number() over (
            partition by
                regexp_replace(lower(title), '[^a-z0-9]', '', 'g'),
                regexp_replace(lower(company), '[^a-z0-9]', '', 'g')
            order by published_at desc nulls last, posting_id
        ) as rn
    from by_id
    where rn = 1
)

select
    posting_id,
    source,
    trim(title) as title,
    trim(company) as company,
    location,
    remote,
    url,
    try_cast(published_at as timestamptz) as published_at,
    tags,
    description,
    lower(title || ' ' || coalesce(tags, '') || ' ' || description) as search_text,
    length(description) as description_chars,
    lang,
    salary_text,
    salary_min,
    salary_max,
    salary_currency
from by_title_company
where rn = 1
