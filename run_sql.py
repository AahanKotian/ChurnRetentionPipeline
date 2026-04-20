import duckdb, sqlite3, pandas as pd

SQLITE_DB = "data/churn.db"
OUT_CSV   = "data/customer_features.csv"

sq = sqlite3.connect(SQLITE_DB)
df_cust = pd.read_sql("SELECT * FROM customers", sq, parse_dates=["signup_date","contract_end_date"])
df_evt  = pd.read_sql("SELECT * FROM events",    sq, parse_dates=["event_date"])
sq.close()

con = duckdb.connect()
# materialise as real tables
con.execute("CREATE TABLE customers AS SELECT * FROM df_cust")
con.execute("CREATE TABLE events    AS SELECT * FROM df_evt")

print(f"customers: {len(df_cust):,}  |  events: {len(df_evt):,}")

con.execute("""
CREATE OR REPLACE VIEW customer_features AS
WITH raw AS (
    SELECT
        c.customer_id, c.plan_type, c.signup_date, c.contract_end_date,
        COUNT(e.event_id)               AS total_events,
        MAX(e.event_date)               AS last_active_date,
        COUNT(DISTINCT e.feature_used)  AS features_used,
        SUM(e.session_minutes)          AS total_session_mins,
        c.churned
    FROM customers c
    LEFT JOIN events e
        ON  c.customer_id = e.customer_id
        AND e.event_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY c.customer_id, c.plan_type, c.signup_date, c.contract_end_date, c.churned
)
SELECT
    customer_id, plan_type,
    COALESCE(total_events, 0)       AS total_events,
    COALESCE(features_used, 0)      AS features_used,
    COALESCE(total_session_mins, 0) AS total_session_mins,
    COALESCE(DATE_DIFF('day', last_active_date::DATE, CURRENT_DATE), 90) AS days_since_active,
    ROUND(COALESCE(total_events, 0) / 12.86, 2)                         AS weekly_event_rate,
    DATE_DIFF('day', signup_date::DATE, CURRENT_DATE)                   AS tenure_days,
    COALESCE(DATE_DIFF('day', CURRENT_DATE, contract_end_date::DATE), -1) AS days_to_renewal,
    ROUND(COALESCE(CAST(features_used AS DOUBLE) / NULLIF(total_events,0), 0), 3) AS feature_breadth,
    churned
FROM raw
""")

con.execute(f"COPY customer_features TO '{OUT_CSV}' (HEADER, DELIMITER ',')")
rows  = con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
churn = con.execute("SELECT AVG(churned) FROM customer_features").fetchone()[0]
print(f"Exported {rows:,} rows  |  churn rate: {churn:.1%}")
print(f"CSV → {OUT_CSV}\n")
print(con.execute("SELECT * FROM customer_features LIMIT 5").df().to_string(index=False))
con.close()
