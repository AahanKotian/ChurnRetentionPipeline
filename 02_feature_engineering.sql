-- ============================================================
-- 02_feature_engineering.sql
-- Build behavioral feature set from raw customer extract
-- Run after 01_extract_customers.sql (or replace CTE with that query)
-- ============================================================

WITH base AS (
    -- Paste or reference output of 01_extract_customers.sql here
    SELECT * FROM customer_raw_extract
),
features AS (
    SELECT
        customer_id,
        plan_type,
        total_events,
        features_used,
        total_session_mins,

        -- Days since last login (inactivity signal)
        DATE_DIFF('day', last_active_date, CURRENT_DATE)
            AS days_since_active,

        -- Average events per week over the 90-day window
        ROUND(total_events / 12.86, 2)
            AS weekly_event_rate,

        -- Customer tenure in days
        DATE_DIFF('day', signup_date, CURRENT_DATE)
            AS tenure_days,

        -- Days until contract renewal (negative = already expired)
        DATE_DIFF('day', CURRENT_DATE, contract_end_date)
            AS days_to_renewal,

        -- Feature breadth: how many distinct features used per event
        ROUND(features_used::FLOAT / NULLIF(total_events, 0), 3)
            AS feature_breadth,

        churned  -- target label
    FROM base
)
SELECT * FROM features;
