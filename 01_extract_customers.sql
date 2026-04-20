-- ============================================================
-- 01_extract_customers.sql
-- Extract raw customer behavior over the last 90 days
-- ============================================================

SELECT
    c.customer_id,
    c.plan_type,
    c.signup_date,
    c.contract_end_date,
    COUNT(e.event_id)               AS total_events,
    MAX(e.event_date)               AS last_active_date,
    MIN(e.event_date)               AS first_event_date,
    COUNT(DISTINCT e.feature_used)  AS features_used,
    SUM(e.session_minutes)          AS total_session_mins,
    c.churned                       -- target label (1 = churned, 0 = retained)
FROM customers c
LEFT JOIN events e
    ON  c.customer_id = e.customer_id
    AND e.event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY
    c.customer_id,
    c.plan_type,
    c.signup_date,
    c.contract_end_date,
    c.churned
ORDER BY c.customer_id;
