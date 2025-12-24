# app.py — VA Empire Tracker (Pro-Pro) with Account Creation + Per-User Private Data
# - Create account / Log in / Log out
# - Each user gets their own SQLite DB: data/user_<id>.db
# - Each user gets their own receipts folder: uploads/user_<id>/
# - WAL + busy_timeout + retry-on-locked to reduce SQLite lock errors
#
# Requirements (requirements.txt):
# streamlit
# pandas
# numpy
# python-dateutil
# bcrypt

import os
import io
import time
import shutil
import sqlite3
import zipfile
from datetime import date, datetime

import CryptContext
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

APP_TITLE = "VA Empire Tracker — Pro-Pro"
AUTH_DB_PATH = "auth.db"
DATA_DIR = "data"
UPLOADS_BASE_DIR = "uploads"

# Lock-handling (WAL + busy_timeout + retries)
DB_BUSY_TIMEOUT_MS = 8000
DB_CONNECT_TIMEOUT_S = 10
DB_WRITE_RETRIES = 10
DB_WRITE_RETRY_SLEEP_S = 0.25


# ============================
# Low-level SQLite helpers
# ============================
def _exec_with_retry(cur, sql, params=()):
    last_err = None
    for _ in range(DB_WRITE_RETRIES):
        try:
            cur.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            last_err = e
            if "database is locked" in str(e).lower():
                time.sleep(DB_WRITE_RETRY_SLEEP_S)
                continue
            raise
    raise last_err


def _apply_sqlite_pragmas(cur):
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute(f"PRAGMA busy_timeout = {DB_BUSY_TIMEOUT_MS};")
    cur.execute("PRAGMA synchronous = NORMAL;")


# ============================
# Auth DB (accounts)
# ============================
def auth_db():
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False, timeout=DB_CONNECT_TIMEOUT_S)
    cur = conn.cursor()
    _apply_sqlite_pragmas(cur)
    return conn


def init_auth_db():
    conn = auth_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                pw_hash BLOB NOT NULL,
                created_at TEXT DEFAULT ''
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_user(email: str, password: str):
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("Enter a valid email.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    conn = auth_db()
    cur = conn.cursor()
    try:
        _exec_with_retry(
            cur,
            "INSERT INTO users (email, pw_hash, created_at) VALUES (?,?,?)",
            (email, pw_hash, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("That email is already registered. Please log in.")
    finally:
        conn.close()


def verify_user(email: str, password: str):
    email = (email or "").strip().lower()
    conn = auth_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, pw_hash FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        if not row:
            return None
        uid, pw_hash = int(row[0]), row[1]
        if bcrypt.checkpw(password.encode("utf-8"), pw_hash):
            return uid
        return None
    finally:
        conn.close()


# ============================
# Per-user paths
# ============================
def user_db_path(user_id: int) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"user_{int(user_id)}.db")


def user_upload_dir(user_id: int) -> str:
    os.makedirs(UPLOADS_BASE_DIR, exist_ok=True)
    path = os.path.join(UPLOADS_BASE_DIR, f"user_{int(user_id)}")
    os.makedirs(path, exist_ok=True)
    return path


# ============================
# Main App DB (per-user)
# ============================
def db():
    if "user_id" not in st.session_state or not st.session_state.user_id:
        raise RuntimeError("Not authenticated")
    path = user_db_path(st.session_state.user_id)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=DB_CONNECT_TIMEOUT_S)
    cur = conn.cursor()
    _apply_sqlite_pragmas(cur)
    return conn


def exec_sql(sql, params=()):
    conn = db()
    cur = conn.cursor()
    try:
        _exec_with_retry(cur, sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def qdf(sql, params=()):
    conn = db()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def get_columns(conn, table: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return {r[1] for r in cur.fetchall()}


def ensure_column(conn, table: str, col: str, col_def: str):
    cols = get_columns(conn, table)
    if col not in cols:
        cur = conn.cursor()
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def};")


def init_db_and_migrate():
    conn = db()
    cur = conn.cursor()
    try:
        # properties
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nickname TEXT NOT NULL,
                address TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'PRIMARY',
                purchase_date TEXT DEFAULT '',
                purchase_price REAL DEFAULT 0,
                est_value REAL DEFAULT 0,
                cash_invested REAL DEFAULT 0,
                reserve_balance REAL DEFAULT 0,
                notes TEXT DEFAULT ''
            );
            """
        )
        for col, col_def in [
            ("address", "address TEXT DEFAULT ''"),
            ("status", "status TEXT NOT NULL DEFAULT 'PRIMARY'"),
            ("purchase_date", "purchase_date TEXT DEFAULT ''"),
            ("purchase_price", "purchase_price REAL DEFAULT 0"),
            ("est_value", "est_value REAL DEFAULT 0"),
            ("cash_invested", "cash_invested REAL DEFAULT 0"),
            ("reserve_balance", "reserve_balance REAL DEFAULT 0"),
            ("notes", "notes TEXT DEFAULT ''"),
        ]:
            ensure_column(conn, "properties", col, col_def)

        # loans
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS loans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER NOT NULL,
                loan_type TEXT NOT NULL DEFAULT 'VA',
                original_balance REAL DEFAULT 0,
                current_balance REAL NOT NULL,
                interest_rate REAL NOT NULL,
                term_months INTEGER NOT NULL,
                start_date TEXT DEFAULT '',
                escrow_monthly REAL DEFAULT 0,
                funding_fee REAL DEFAULT 0,
                default_extra REAL DEFAULT 0,
                FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE
            );
            """
        )
        for col, col_def in [
            ("loan_type", "loan_type TEXT NOT NULL DEFAULT 'VA'"),
            ("original_balance", "original_balance REAL DEFAULT 0"),
            ("start_date", "start_date TEXT DEFAULT ''"),
            ("escrow_monthly", "escrow_monthly REAL DEFAULT 0"),
            ("funding_fee", "funding_fee REAL DEFAULT 0"),
            ("default_extra", "default_extra REAL DEFAULT 0"),
        ]:
            ensure_column(conn, "loans", col, col_def)

        # rentals
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rentals (
                property_id INTEGER PRIMARY KEY,
                rent_monthly REAL DEFAULT 0,
                vacancy_rate REAL DEFAULT 5,
                management_rate REAL DEFAULT 0,
                maintenance_rate REAL DEFAULT 5,
                capex_rate REAL DEFAULT 5,
                taxes_monthly REAL DEFAULT 0,
                insurance_monthly REAL DEFAULT 0,
                hoa_monthly REAL DEFAULT 0,
                utilities_monthly REAL DEFAULT 0,
                other_monthly REAL DEFAULT 0,
                FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE
            );
            """
        )

        # settings
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                county_loan_limit REAL DEFAULT 0,
                reserve_months_target REAL DEFAULT 3,
                use_primary_first INTEGER DEFAULT 1,
                appreciation_rate_annual REAL DEFAULT 3,
                include_reserve_lock INTEGER DEFAULT 1
            );
            """
        )
        for col, col_def in [
            ("county_loan_limit", "county_loan_limit REAL DEFAULT 0"),
            ("reserve_months_target", "reserve_months_target REAL DEFAULT 3"),
            ("use_primary_first", "use_primary_first INTEGER DEFAULT 1"),
            ("appreciation_rate_annual", "appreciation_rate_annual REAL DEFAULT 3"),
            ("include_reserve_lock", "include_reserve_lock INTEGER DEFAULT 1"),
        ]:
            ensure_column(conn, "settings", col, col_def)

        # tenants
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                phone TEXT DEFAULT '',
                email TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            );
            """
        )

        # leases
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                rent_monthly REAL NOT NULL DEFAULT 0,
                security_deposit REAL DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                notes TEXT DEFAULT '',
                FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE,
                FOREIGN KEY(tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT
            );
            """
        )

        # transactions
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER NOT NULL,
                tx_date TEXT NOT NULL,
                tx_type TEXT NOT NULL,      -- INCOME / EXPENSE
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                memo TEXT DEFAULT '',
                created_at TEXT DEFAULT '',
                FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE
            );
            """
        )

        # attachments
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                uploaded_at TEXT DEFAULT '',
                FOREIGN KEY(transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
            );
            """
        )

        # vacancies
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vacancies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE
            );
            """
        )

        _exec_with_retry(cur, "INSERT OR IGNORE INTO settings (id) VALUES (1);", ())
        _exec_with_retry(
            cur,
            """
            UPDATE settings
            SET
              county_loan_limit = COALESCE(county_loan_limit, 0),
              reserve_months_target = COALESCE(reserve_months_target, 3),
              use_primary_first = COALESCE(use_primary_first, 1),
              appreciation_rate_annual = COALESCE(appreciation_rate_annual, 3),
              include_reserve_lock = COALESCE(include_reserve_lock, 1)
            WHERE id = 1;
            """,
            (),
        )

        conn.commit()
    finally:
        conn.close()


# ============================
# Safe deletes
# ============================
def safe_delete_property(property_id: int):
    conn = db()
    cur = conn.cursor()
    try:
        cur.execute("BEGIN;")
        _exec_with_retry(
            cur,
            "DELETE FROM attachments WHERE transaction_id IN (SELECT id FROM transactions WHERE property_id = ?)",
            (property_id,),
        )
        _exec_with_retry(cur, "DELETE FROM transactions WHERE property_id = ?", (property_id,))
        _exec_with_retry(cur, "DELETE FROM vacancies WHERE property_id = ?", (property_id,))
        _exec_with_retry(cur, "DELETE FROM leases WHERE property_id = ?", (property_id,))
        _exec_with_retry(cur, "DELETE FROM rentals WHERE property_id = ?", (property_id,))
        _exec_with_retry(cur, "DELETE FROM loans WHERE property_id = ?", (property_id,))
        _exec_with_retry(cur, "DELETE FROM properties WHERE id = ?", (property_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def safe_delete_tenant(tenant_id: int, delete_leases: bool = False):
    conn = db()
    cur = conn.cursor()
    try:
        cur.execute("BEGIN;")
        cur.execute("SELECT COUNT(*) FROM leases WHERE tenant_id = ?", (tenant_id,))
        lease_count = int(cur.fetchone()[0] or 0)

        if lease_count > 0 and not delete_leases:
            raise ValueError(f"Tenant has {lease_count} lease(s). Check 'Delete tenant leases too' to proceed.")

        if lease_count > 0 and delete_leases:
            _exec_with_retry(cur, "DELETE FROM leases WHERE tenant_id = ?", (tenant_id,))

        _exec_with_retry(cur, "DELETE FROM tenants WHERE id = ?", (tenant_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================
# Finance helpers
# ============================
def monthly_pi(balance, annual_rate_pct, term_months):
    r = (annual_rate_pct / 100) / 12
    if term_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return float(balance) / float(term_months)
    return balance * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)


def rental_cashflow_estimate(rental_row: dict, monthly_piti_escrow: float):
    rent = float(rental_row.get("rent_monthly", 0) or 0)
    vacancy = float(rental_row.get("vacancy_rate", 0) or 0) / 100
    mgmt = float(rental_row.get("management_rate", 0) or 0) / 100
    maint = float(rental_row.get("maintenance_rate", 0) or 0) / 100
    capex = float(rental_row.get("capex_rate", 0) or 0) / 100

    fixed = sum(
        [
            float(rental_row.get("taxes_monthly", 0) or 0),
            float(rental_row.get("insurance_monthly", 0) or 0),
            float(rental_row.get("hoa_monthly", 0) or 0),
            float(rental_row.get("utilities_monthly", 0) or 0),
            float(rental_row.get("other_monthly", 0) or 0),
        ]
    )

    effective_rent = rent * (1 - vacancy)
    variable = rent * (mgmt + maint + capex)
    return effective_rent - variable - fixed - float(monthly_piti_escrow)


def compute_debt_service_monthly(loans_df_for_property: pd.DataFrame) -> float:
    if loans_df_for_property.empty:
        return 0.0
    total = 0.0
    for _, l in loans_df_for_property.iterrows():
        pi = monthly_pi(float(l["current_balance"]), float(l["interest_rate"]), int(l["term_months"]))
        total += pi + float(l.get("escrow_monthly", 0) or 0)
    return total


def property_metrics(pid: int, props_df, loans_df, rentals_df):
    p = props_df[props_df["id"] == pid].iloc[0].to_dict()
    r = rentals_df[rentals_df["property_id"] == pid]
    l = loans_df[loans_df["property_id"] == pid]

    value = float(p.get("est_value", 0) or 0)
    if value <= 0:
        value = float(p.get("purchase_price", 0) or 0)

    debt_service_m = compute_debt_service_monthly(l)
    rent_row = r.iloc[0].to_dict() if not r.empty else {}
    cashflow_m = rental_cashflow_estimate(rent_row, debt_service_m) if rent_row else 0.0

    rent = float(rent_row.get("rent_monthly", 0) or 0)
    vacancy = float(rent_row.get("vacancy_rate", 0) or 0) / 100
    mgmt = float(rent_row.get("management_rate", 0) or 0) / 100
    maint = float(rent_row.get("maintenance_rate", 0) or 0) / 100
    capex = float(rent_row.get("capex_rate", 0) or 0) / 100

    fixed = sum(
        [
            float(rent_row.get("taxes_monthly", 0) or 0),
            float(rent_row.get("insurance_monthly", 0) or 0),
            float(rent_row.get("hoa_monthly", 0) or 0),
            float(rent_row.get("utilities_monthly", 0) or 0),
            float(rent_row.get("other_monthly", 0) or 0),
        ]
    )

    noi_m = (rent * (1 - vacancy)) - (rent * (mgmt + maint + capex)) - fixed
    noi_a = noi_m * 12

    cap_rate = (noi_a / value) if value > 0 else 0.0
    dscr = (noi_m / debt_service_m) if debt_service_m > 0 else 0.0

    cash_invested = float(p.get("cash_invested", 0) or 0)
    coc = ((cashflow_m * 12) / cash_invested) if cash_invested > 0 else 0.0

    return {
        "Value_Used": value,
        "Debt_Service_Monthly": debt_service_m,
        "NOI_Annual": noi_a,
        "Cap_Rate": cap_rate,
        "DSCR": dscr,
        "Cashflow_Monthly": cashflow_m,
        "Cash_on_Cash": coc,
    }


def compute_required_reserves(props_df, loans_df, rentals_df, reserve_months_target: float) -> float:
    req = 0.0
    for _, p in props_df.iterrows():
        pid = int(p["id"])
        l = loans_df[loans_df["property_id"] == pid]
        if l.empty:
            continue
        debt = compute_debt_service_monthly(l)

        if p["status"] == "PRIMARY":
            outflow = debt
        else:
            r = rentals_df[rentals_df["property_id"] == pid]
            fixed = 0.0
            if not r.empty:
                rr = r.iloc[0].to_dict()
                fixed = sum(
                    [
                        float(rr.get("taxes_monthly", 0) or 0),
                        float(rr.get("insurance_monthly", 0) or 0),
                        float(rr.get("hoa_monthly", 0) or 0),
                        float(rr.get("utilities_monthly", 0) or 0),
                        float(rr.get("other_monthly", 0) or 0),
                    ]
                )
            outflow = debt + fixed

        req += float(reserve_months_target) * float(outflow)
    return req


# ============================
# Net Worth Timeline (Upgraded)
# ============================
def debt_path_with_snowball(
    loans_df: pd.DataFrame,
    props_df: pd.DataFrame,
    rentals_df: pd.DataFrame,
    extra_base_monthly: float,
    include_rental_routing: bool,
    primary_first: bool,
    reserve_lock: bool,
    reserve_months_target: float,
    months: int = 240,
):
    if loans_df.empty:
        return np.zeros(months), 0.0, 0.0, 0.0

    routed_cf = 0.0
    if include_rental_routing and not props_df.empty and not rentals_df.empty:
        for _, p in props_df.iterrows():
            if p["status"] != "RENTAL":
                continue
            pid = int(p["id"])
            l = loans_df[loans_df["property_id"] == pid]
            r = rentals_df[rentals_df["property_id"] == pid]
            if l.empty or r.empty:
                continue
            debt = compute_debt_service_monthly(l)
            cf = rental_cashflow_estimate(r.iloc[0].to_dict(), debt)
            routed_cf += max(cf, 0.0)

    extra_allowed = float(extra_base_monthly) + float(routed_cf)

    if reserve_lock:
        total_reserves = float(props_df["reserve_balance"].fillna(0).sum()) if not props_df.empty else 0.0
        required = compute_required_reserves(props_df, loans_df, rentals_df, float(reserve_months_target))
        if required > total_reserves + 1e-6:
            extra_allowed = 0.0

    prop_status = dict(zip(props_df["id"], props_df["status"])) if not props_df.empty else {}
    loans = loans_df.copy()
    loans["status"] = loans["property_id"].map(lambda pid: prop_status.get(pid, "RENTAL"))
    loans["min_pi"] = loans.apply(
        lambda r: monthly_pi(r["current_balance"], r["interest_rate"], int(r["term_months"])),
        axis=1,
    )
    loans = loans.reset_index(drop=True)

    balances = {int(r["id"]): float(r["current_balance"]) for _, r in loans.iterrows()}
    rates = {int(r["id"]): float(r["interest_rate"]) / 100 / 12 for _, r in loans.iterrows()}
    mins = {int(r["id"]): float(r["min_pi"]) for _, r in loans.iterrows()}

    def sort_key(lid):
        row = loans[loans["id"] == lid].iloc[0]
        status = row["status"]
        primary_rank = 0 if (primary_first and status == "PRIMARY") else 1
        return (primary_rank, -float(row["interest_rate"]), float(balances[lid]))

    series = []
    for _m in range(months):
        remaining = [lid for lid, b in balances.items() if b > 0.01]
        if not remaining:
            series.append(0.0)
            continue

        target = sorted(remaining, key=sort_key)[0]

        for lid in remaining:
            b = balances[lid]
            i = b * rates[lid]
            principal = max(mins[lid] - i, 0.0)
            extra = float(extra_allowed) if lid == target else 0.0
            pay_principal = principal + extra
            if pay_principal > b:
                pay_principal = b
            balances[lid] = b - pay_principal

        series.append(sum(max(v, 0.0) for v in balances.values()))

    return np.array(series), float(extra_base_monthly), float(routed_cf), float(extra_allowed)


def project_networth_over_time(
    props_df: pd.DataFrame,
    loans_df: pd.DataFrame,
    rentals_df: pd.DataFrame,
    appreciation_rate_annual_pct: float,
    months: int,
    extra_base_monthly: float,
    include_rental_routing: bool,
    primary_first: bool,
    reserve_lock: bool,
    reserve_months_target: float,
):
    start_total_value = 0.0
    if not props_df.empty:
        for _, p in props_df.iterrows():
            v = float(p.get("est_value", 0) or 0)
            if v <= 0:
                v = float(p.get("purchase_price", 0) or 0)
            start_total_value += v

    debt_series, base_extra, routed_cf, extra_allowed = debt_path_with_snowball(
        loans_df=loans_df,
        props_df=props_df,
        rentals_df=rentals_df,
        extra_base_monthly=float(extra_base_monthly),
        include_rental_routing=bool(include_rental_routing),
        primary_first=bool(primary_first),
        reserve_lock=bool(reserve_lock),
        reserve_months_target=float(reserve_months_target),
        months=int(months),
    )

    r_m = (float(appreciation_rate_annual_pct) / 100.0) / 12.0
    value_series = np.array([start_total_value * ((1 + r_m) ** i) for i in range(int(months))])

    start_dt = date.today().replace(day=1)
    dates = [start_dt + relativedelta(months=i) for i in range(int(months))]
    networth = value_series - debt_series

    df = pd.DataFrame(
        {
            "Date": dates,
            "Portfolio_Value": value_series,
            "Debt": debt_series,
            "Net_Worth": networth,
        }
    )
    meta = {"Extra_Base": base_extra, "Routed_Rental_CF": routed_cf, "Extra_Allowed": extra_allowed}
    return df, meta


# ============================
# Backup / Restore (per-user)
# ============================
def build_backup_zip_bytes(user_db: str, user_uploads: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(user_db):
            z.write(user_db, arcname="user.db")
        if os.path.isdir(user_uploads):
            for root, _, files in os.walk(user_uploads):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, start=user_uploads)
                    z.write(full, arcname=os.path.join("uploads", rel))
    buf.seek(0)
    return buf.read()


def safe_restore_from_zip(zip_bytes: bytes, user_db: str, user_uploads: str):
    tmp_dir = "_restore_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(tmp_dir)

    extracted_db = os.path.join(tmp_dir, "user.db")
    extracted_uploads = os.path.join(tmp_dir, "uploads")

    if not os.path.exists(extracted_db):
        raise ValueError("Backup ZIP missing user.db")

    # Replace DB
    os.makedirs(os.path.dirname(user_db), exist_ok=True)
    if os.path.exists(user_db):
        os.replace(user_db, user_db + ".bak")
    os.replace(extracted_db, user_db)

    # Replace uploads
    os.makedirs(user_uploads, exist_ok=True)
    if os.path.exists(user_uploads):
        shutil.rmtree(user_uploads)
    if os.path.isdir(extracted_uploads):
        shutil.copytree(extracted_uploads, user_uploads)
    else:
        os.makedirs(user_uploads, exist_ok=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================
# Small UI helpers
# ============================
def parse_iso_date(s: str):
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# ============================
# Streamlit start + AUTH GATE
# ============================
st.set_page_config(page_title=APP_TITLE, layout="wide")
init_auth_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

if not st.session_state.user_id:
    st.title("VA Empire Tracker")
    st.caption("Create an account or log in to access your private dashboard.")

    tab1, tab2 = st.tabs(["Log in", "Create account"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            ok = st.form_submit_button("Log in")
            if ok:
                uid = verify_user(email, pw)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.user_email = (email or "").strip().lower()
                    st.success("Logged in.")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with tab2:
        with st.form("create_form"):
            email = st.text_input("Email", key="create_email")
            pw1 = st.text_input("Password (min 8 chars)", type="password", key="create_pw1")
            pw2 = st.text_input("Confirm password", type="password", key="create_pw2")
            ok = st.form_submit_button("Create account")
            if ok:
                try:
                    if pw1 != pw2:
                        raise ValueError("Passwords do not match.")
                    create_user(email, pw1)
                    uid = verify_user(email, pw1)
                    st.session_state.user_id = uid
                    st.session_state.user_email = (email or "").strip().lower()
                    st.success("Account created. You’re logged in.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.stop()

# Logged-in sidebar
st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")
if st.sidebar.button("Log out"):
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.rerun()

# Init per-user database
init_db_and_migrate()

# Load per-user data
props = qdf("SELECT * FROM properties ORDER BY id DESC")
loans = qdf("SELECT * FROM loans ORDER BY id DESC")
rentals = qdf("SELECT * FROM rentals ORDER BY property_id DESC")
tenants = qdf("SELECT * FROM tenants ORDER BY id DESC")
leases = qdf(
    """
    SELECT l.*, t.full_name AS tenant_name
    FROM leases l
    JOIN tenants t ON t.id = l.tenant_id
    ORDER BY l.id DESC
    """
)
txs = qdf("SELECT * FROM transactions ORDER BY tx_date DESC, id DESC")
attachments = qdf("SELECT * FROM attachments ORDER BY id DESC")
vacancies = qdf("SELECT * FROM vacancies ORDER BY start_date DESC, id DESC")
settings_df = qdf("SELECT * FROM settings WHERE id=1")
settings = (
    settings_df.iloc[0].to_dict()
    if not settings_df.empty
    else {
        "county_loan_limit": 0,
        "reserve_months_target": 3,
        "use_primary_first": 1,
        "appreciation_rate_annual": 3,
        "include_reserve_lock": 1,
    }
)


def prop_label(pid: int) -> str:
    if props.empty:
        return str(pid)
    p = props[props["id"] == pid].iloc[0]
    return f"#{int(p['id'])} — {p['nickname']} ({p['status']})"


# ============================
# Main UI
# ============================
st.title(APP_TITLE)
st.caption("Private per-user data (accounts) • SQLite WAL lock fixes • Pro-Pro features")

page = st.sidebar.radio(
    "Navigate",
    [
        "Empire Dashboard",
        "Properties (Add / Edit / Delete / Move→Rent)",
        "Loans",
        "Rentals",
        "Tenants & Leases (Pro)",
        "Ledger & Receipts (Pro)",
        "Vacancy Tracker (Pro-Pro)",
        "Reports (Rent Roll + Tax) (Pro-Pro)",
        "Net Worth Timeline (Upgraded)",
        "Backup & Restore (Pro-Pro)",
        "Exports",
        "Settings",
    ],
)

# ============================
# Pages
# ============================
if page == "Empire Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    total_balance = float(loans["current_balance"].sum()) if not loans.empty else 0.0
    total_props = int(len(props))

    total_value = float(props["est_value"].fillna(0).sum()) if not props.empty else 0.0
    if total_value <= 0:
        total_value = float(props["purchase_price"].fillna(0).sum()) if not props.empty else 0.0

    equity_est = max(total_value - total_balance, 0.0) if total_value > 0 else 0.0

    cashflows = []
    if not props.empty:
        for _, p in props.iterrows():
            if p["status"] != "RENTAL":
                continue
            pid = int(p["id"])
            cashflows.append(property_metrics(pid, props, loans, rentals)["Cashflow_Monthly"])
    est_rental_cashflow = float(np.sum(cashflows)) if cashflows else 0.0

    c1.metric("Properties", total_props)
    c2.metric("Total Loan Debt", f"${total_balance:,.0f}")
    c3.metric("Est. Net Cashflow (rentals)", f"${est_rental_cashflow:,.0f}/mo")
    c4.metric("Est. Equity", f"${equity_est:,.0f}")

    st.divider()
    st.subheader("Reminders")
    days = st.slider("Show leases expiring within (days)", 7, 180, 45, 1)
    expiring_rows = []
    today = date.today()

    if not leases.empty:
        for _, l in leases.iterrows():
            if int(l.get("is_active", 1)) != 1:
                continue
            end_dt = parse_iso_date(l["end_date"])
            if end_dt is None:
                continue
            delta = (end_dt - today).days
            if 0 <= delta <= days:
                expiring_rows.append(
                    {
                        "Lease_ID": int(l["id"]),
                        "Property": prop_label(int(l["property_id"])),
                        "Tenant": l["tenant_name"],
                        "End_Date": str(end_dt),
                        "Days_Left": delta,
                        "Rent_Monthly": float(l["rent_monthly"] or 0),
                    }
                )

    if expiring_rows:
        st.warning("Leases expiring soon")
        st.dataframe(pd.DataFrame(expiring_rows).sort_values("Days_Left"), use_container_width=True)
    else:
        st.success("No leases expiring in that window.")

    st.divider()
    st.subheader("Latest Ledger Activity")
    if txs.empty:
        st.info("No transactions yet.")
    else:
        st.dataframe(txs.head(20), use_container_width=True)


elif page == "Properties (Add / Edit / Delete / Move→Rent)":
    st.subheader("Add a Property")
    with st.form("add_property"):
        nickname = st.text_input("Nickname", "")
        address = st.text_input("Address (optional)", "")
        status = st.selectbox("Status", ["PRIMARY", "RENTAL"])
        purchase_date = st.date_input("Purchase date", value=date.today())
        purchase_price = st.number_input("Purchase price", min_value=0.0, step=1000.0, value=0.0)
        est_value = st.number_input("Estimated current value (optional)", min_value=0.0, step=1000.0, value=0.0)
        cash_invested = st.number_input("Cash invested (for Cash-on-Cash)", min_value=0.0, step=500.0, value=0.0)
        reserve_balance = st.number_input("Reserve balance", min_value=0.0, step=500.0, value=0.0)
        notes = st.text_area("Notes", "")
        submitted = st.form_submit_button("Add Property")
        if submitted:
            if not nickname.strip():
                st.error("Nickname is required.")
            else:
                exec_sql(
                    """
                    INSERT INTO properties (nickname, address, status, purchase_date, purchase_price, est_value, cash_invested, reserve_balance, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        nickname.strip(),
                        address.strip(),
                        status,
                        purchase_date.isoformat(),
                        float(purchase_price),
                        float(est_value),
                        float(cash_invested),
                        float(reserve_balance),
                        notes.strip(),
                    ),
                )
                st.success("Property added.")
                st.rerun()

    st.divider()
    st.subheader("Edit / Delete / Move→Rent")
    if props.empty:
        st.info("No properties yet.")
    else:
        chosen_pid = st.selectbox("Select property", props["id"].tolist(), format_func=prop_label)
        p = props[props["id"] == chosen_pid].iloc[0]

        colA, colB = st.columns([2, 1])
        with colA:
            new_nickname = st.text_input("Nickname", value=p["nickname"])
            new_address = st.text_input("Address", value=p.get("address", "") or "")
            new_status = st.selectbox("Status", ["PRIMARY", "RENTAL"], index=0 if p["status"] == "PRIMARY" else 1)
            new_purchase_price = st.number_input("Purchase price", min_value=0.0, step=1000.0, value=float(p.get("purchase_price", 0) or 0))
            new_est_value = st.number_input("Estimated current value", min_value=0.0, step=1000.0, value=float(p.get("est_value", 0) or 0))
            new_cash_invested = st.number_input("Cash invested", min_value=0.0, step=500.0, value=float(p.get("cash_invested", 0) or 0))
            new_reserve_balance = st.number_input("Reserve balance", min_value=0.0, step=500.0, value=float(p.get("reserve_balance", 0) or 0))
            new_notes = st.text_area("Notes", value=p.get("notes", "") or "")

            if st.button("Save Changes"):
                exec_sql(
                    """
                    UPDATE properties
                    SET nickname=?, address=?, status=?, purchase_price=?, est_value=?, cash_invested=?, reserve_balance=?, notes=?
                    WHERE id=?
                    """,
                    (
                        new_nickname.strip(),
                        new_address.strip(),
                        new_status,
                        float(new_purchase_price),
                        float(new_est_value),
                        float(new_cash_invested),
                        float(new_reserve_balance),
                        new_notes.strip(),
                        int(chosen_pid),
                    ),
                )
                st.success("Saved.")
                st.rerun()

            st.divider()
            st.subheader("One-click Move → Rent")
            st.caption("Flips PRIMARY → RENTAL and initializes a rental row if missing.")
            if st.button("Move → Rent"):
                exec_sql("UPDATE properties SET status='RENTAL' WHERE id=?", (int(chosen_pid),))
                existing = qdf("SELECT * FROM rentals WHERE property_id=?", (int(chosen_pid),))
                if existing.empty:
                    exec_sql("INSERT INTO rentals (property_id) VALUES (?)", (int(chosen_pid),))
                st.success("Moved to RENTAL. Go to Rentals to enter rent + expenses.")
                st.rerun()

        with colB:
            st.warning("Delete House (safe delete)")
            st.caption("Deletes property + loans + rentals + leases + ledger + receipts + vacancies.")
            confirm = st.text_input("Type DELETE to confirm", value="", key="delete_confirm_prop")
            if st.button("Delete Property Now"):
                if confirm.strip().upper() != "DELETE":
                    st.error("Type DELETE to confirm.")
                else:
                    try:
                        safe_delete_property(int(chosen_pid))
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")


elif page == "Loans":
    st.subheader("Add a Loan")
    if props.empty:
        st.info("Add a property first.")
    else:
        with st.form("add_loan"):
            pid = st.selectbox("Property", props["id"].tolist(), format_func=prop_label)
            loan_type = st.selectbox("Loan type", ["VA", "CONV", "FHA", "OTHER"])
            original_balance = st.number_input("Original balance", min_value=0.0, step=1000.0, value=0.0)
            current_balance = st.number_input("Current balance", min_value=0.0, step=1000.0, value=0.0)
            interest_rate = st.number_input("Interest rate (%)", min_value=0.0, step=0.125, value=5.0)
            term_months = st.number_input("Term (months)", min_value=1, step=12, value=360)
            start_date = st.date_input("Loan start date", value=date.today())
            escrow_monthly = st.number_input("Escrow estimate (monthly)", min_value=0.0, step=50.0, value=0.0)
            default_extra = st.number_input("Default extra principal (monthly)", min_value=0.0, step=50.0, value=0.0)
            submitted = st.form_submit_button("Add Loan")
            if submitted:
                if current_balance <= 0:
                    st.error("Current balance must be > 0.")
                else:
                    exec_sql(
                        """
                        INSERT INTO loans (property_id, loan_type, original_balance, current_balance, interest_rate,
                                           term_months, start_date, escrow_monthly, default_extra)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(pid),
                            loan_type,
                            float(original_balance),
                            float(current_balance),
                            float(interest_rate),
                            int(term_months),
                            start_date.isoformat(),
                            float(escrow_monthly),
                            float(default_extra),
                        ),
                    )
                    st.success("Loan added.")
                    st.rerun()

    st.divider()
    st.subheader("Loans")
    if loans.empty:
        st.info("No loans yet.")
    else:
        st.dataframe(loans, use_container_width=True)


elif page == "Rentals":
    st.subheader("Rental Settings")
    if props.empty:
        st.info("Add a property first.")
    else:
        pid = st.selectbox("Select property", props["id"].tolist(), format_func=prop_label)
        existing = rentals[rentals["property_id"] == pid]

        d = {
            "rent_monthly": 0.0,
            "vacancy_rate": 5.0,
            "management_rate": 0.0,
            "maintenance_rate": 5.0,
            "capex_rate": 5.0,
            "taxes_monthly": 0.0,
            "insurance_monthly": 0.0,
            "hoa_monthly": 0.0,
            "utilities_monthly": 0.0,
            "other_monthly": 0.0,
        }
        if not existing.empty:
            row = existing.iloc[0].to_dict()
            for k in d:
                d[k] = float(row.get(k, d[k]) or d[k])

        with st.form("rental_form"):
            rent = st.number_input("Rent (monthly)", min_value=0.0, step=50.0, value=d["rent_monthly"])
            c1, c2, c3 = st.columns(3)
            vacancy = c1.number_input("Vacancy (%)", min_value=0.0, max_value=50.0, step=0.5, value=d["vacancy_rate"])
            mgmt = c2.number_input("Management (%)", min_value=0.0, max_value=25.0, step=0.5, value=d["management_rate"])
            maint = c3.number_input("Maintenance (%)", min_value=0.0, max_value=25.0, step=0.5, value=d["maintenance_rate"])
            capex = st.number_input("CapEx reserve (%)", min_value=0.0, max_value=25.0, step=0.5, value=d["capex_rate"])

            c4, c5, c6 = st.columns(3)
            taxes_m = c4.number_input("Taxes (monthly)", min_value=0.0, step=25.0, value=d["taxes_monthly"])
            ins_m = c5.number_input("Insurance (monthly)", min_value=0.0, step=25.0, value=d["insurance_monthly"])
            hoa_m = c6.number_input("HOA (monthly)", min_value=0.0, step=25.0, value=d["hoa_monthly"])

            c7, c8 = st.columns(2)
            utils_m = c7.number_input("Utilities you pay (monthly)", min_value=0.0, step=25.0, value=d["utilities_monthly"])
            other_m = c8.number_input("Other fixed (monthly)", min_value=0.0, step=25.0, value=d["other_monthly"])

            saved = st.form_submit_button("Save rental settings")
            if saved:
                exec_sql(
                    """
                    INSERT INTO rentals (property_id, rent_monthly, vacancy_rate, management_rate, maintenance_rate, capex_rate,
                                        taxes_monthly, insurance_monthly, hoa_monthly, utilities_monthly, other_monthly)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(property_id) DO UPDATE SET
                        rent_monthly=excluded.rent_monthly,
                        vacancy_rate=excluded.vacancy_rate,
                        management_rate=excluded.management_rate,
                        maintenance_rate=excluded.maintenance_rate,
                        capex_rate=excluded.capex_rate,
                        taxes_monthly=excluded.taxes_monthly,
                        insurance_monthly=excluded.insurance_monthly,
                        hoa_monthly=excluded.hoa_monthly,
                        utilities_monthly=excluded.utilities_monthly,
                        other_monthly=excluded.other_monthly;
                    """,
                    (int(pid), rent, vacancy, mgmt, maint, capex, taxes_m, ins_m, hoa_m, utils_m, other_m),
                )
                st.success("Saved.")
                st.rerun()

        st.divider()
        st.subheader("Estimated Metrics Preview")
        if loans[loans["property_id"] == pid].empty:
            st.info("Add a loan to compute debt service / DSCR.")
        else:
            m = property_metrics(int(pid), props, loans, rentals)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("NOI (Annual)", f"${m['NOI_Annual']:,.0f}")
            c2.metric("Cap Rate", f"{m['Cap_Rate']*100:,.2f}%")
            c3.metric("DSCR", f"{m['DSCR']:,.2f}")
            c4.metric("Cashflow (Monthly)", f"${m['Cashflow_Monthly']:,.0f}")


elif page == "Tenants & Leases (Pro)":
    st.subheader("Tenants")

    with st.expander("Add Tenant", expanded=False):
        with st.form("add_tenant"):
            name = st.text_input("Full name")
            phone = st.text_input("Phone (optional)")
            email = st.text_input("Email (optional)")
            notes = st.text_area("Notes (optional)")
            if st.form_submit_button("Add Tenant"):
                if not name.strip():
                    st.error("Name required.")
                else:
                    exec_sql(
                        "INSERT INTO tenants (full_name, phone, email, notes) VALUES (?,?,?,?)",
                        (name.strip(), phone.strip(), email.strip(), notes.strip()),
                    )
                    st.success("Tenant added.")
                    st.rerun()

    if tenants.empty:
        st.info("No tenants yet.")
    else:
        st.dataframe(tenants, use_container_width=True)

        st.divider()
        st.subheader("Delete Tenant (safe)")
        chosen_tid = st.selectbox(
            "Select tenant",
            tenants["id"].tolist(),
            format_func=lambda tid: tenants[tenants["id"] == tid].iloc[0]["full_name"],
        )

        linked = 0
        if not leases.empty:
            linked = int((leases["tenant_id"] == chosen_tid).sum())

        st.write(f"Leases linked to this tenant: **{linked}**")
        delete_leases_too = st.checkbox("Delete tenant leases too (required if leases exist)", value=False)
        confirm_t = st.text_input("Type DELETE to confirm tenant delete", value="", key="delete_confirm_tenant")
        if st.button("Delete Tenant Now"):
            if confirm_t.strip().upper() != "DELETE":
                st.error("Type DELETE to confirm.")
            else:
                try:
                    safe_delete_tenant(int(chosen_tid), delete_leases=bool(delete_leases_too))
                    st.success("Tenant deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Tenant delete failed: {e}")

    st.divider()
    st.subheader("Leases")
    if props.empty or tenants.empty:
        st.info("Add at least one property and one tenant to create a lease.")
    else:
        with st.form("add_lease"):
            pid = st.selectbox("Property", props["id"].tolist(), format_func=prop_label)
            tid = st.selectbox(
                "Tenant",
                tenants["id"].tolist(),
                format_func=lambda x: tenants[tenants["id"] == x].iloc[0]["full_name"],
            )
            start = st.date_input("Lease start", value=date.today())
            end = st.date_input("Lease end", value=date.today() + relativedelta(years=1))
            rent = st.number_input("Rent (monthly)", min_value=0.0, step=50.0, value=0.0)
            dep = st.number_input("Security deposit", min_value=0.0, step=50.0, value=0.0)
            active = st.checkbox("Active lease", value=True)
            notes = st.text_area("Notes (optional)")
            if st.form_submit_button("Create Lease"):
                exec_sql(
                    """
                    INSERT INTO leases (property_id, tenant_id, start_date, end_date, rent_monthly, security_deposit, is_active, notes)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (int(pid), int(tid), start.isoformat(), end.isoformat(), float(rent), float(dep), 1 if active else 0, notes.strip()),
                )
                st.success("Lease created.")
                st.rerun()

    if leases.empty:
        st.info("No leases yet.")
    else:
        st.dataframe(leases, use_container_width=True)


elif page == "Ledger & Receipts (Pro)":
    st.subheader("Add Transaction")
    if props.empty:
        st.info("Add a property first.")
    else:
        with st.form("add_tx"):
            pid = st.selectbox("Property", props["id"].tolist(), format_func=prop_label)
            tx_date = st.date_input("Date", value=date.today())
            tx_type = st.selectbox("Type", ["INCOME", "EXPENSE"])
            category = st.selectbox(
                "Category",
                [
                    "Rent",
                    "Repairs",
                    "CapEx",
                    "Taxes",
                    "Insurance",
                    "HOA",
                    "Utilities",
                    "Management",
                    "Mortgage",
                    "Turnover",
                    "Supplies",
                    "Legal/Professional",
                    "Advertising",
                    "Travel",
                    "Other",
                ],
            )
            amount = st.number_input("Amount", min_value=0.0, step=10.0, value=0.0)
            memo = st.text_input("Memo (optional)")
            receipt = st.file_uploader("Receipt (optional)", type=["png", "jpg", "jpeg", "pdf"])

            if st.form_submit_button("Save Transaction"):
                if amount <= 0:
                    st.error("Amount must be > 0.")
                else:
                    tx_id = exec_sql(
                        """
                        INSERT INTO transactions (property_id, tx_date, tx_type, category, amount, memo, created_at)
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        (
                            int(pid),
                            tx_date.isoformat(),
                            tx_type,
                            category,
                            float(amount),
                            memo.strip(),
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )

                    if receipt is not None:
                        udir = user_upload_dir(st.session_state.user_id)
                        safe_name = f"tx{tx_id}_{receipt.name}".replace("/", "_")
                        path = os.path.join(udir, safe_name)
                        with open(path, "wb") as f:
                            f.write(receipt.getbuffer())
                        exec_sql(
                            "INSERT INTO attachments (transaction_id, filename, filepath, uploaded_at) VALUES (?,?,?,?)",
                            (int(tx_id), receipt.name, path, datetime.now().isoformat(timespec="seconds")),
                        )

                    st.success("Transaction saved.")
                    st.rerun()

    st.divider()
    st.subheader("Ledger")
    if txs.empty:
        st.info("No transactions yet.")
    else:
        col1, col2, col3 = st.columns(3)
        prop_filter = col1.selectbox(
            "Property",
            ["All"] + props["id"].tolist(),
            format_func=lambda x: "All" if x == "All" else prop_label(int(x)),
        )
        type_filter = col2.selectbox("Type", ["All", "INCOME", "EXPENSE"])
        cat_filter = col3.selectbox("Category", ["All"] + sorted(txs["category"].unique().tolist()))

        df = txs.copy()
        if prop_filter != "All":
            df = df[df["property_id"] == int(prop_filter)]
        if type_filter != "All":
            df = df[df["tx_type"] == type_filter]
        if cat_filter != "All":
            df = df[df["category"] == cat_filter]

        st.dataframe(df, use_container_width=True)

        st.subheader("Receipts")
        if attachments.empty:
            st.info("No receipts uploaded yet.")
        else:
            st.dataframe(attachments, use_container_width=True)
            st.caption("Receipts are stored per-user in uploads/user_<id>/.")


elif page == "Vacancy Tracker (Pro-Pro)":
    st.subheader("Vacancy Tracker")
    st.caption("Log vacancy periods. Helpful for real cashflow + year-end reporting.")

    if props.empty:
        st.info("Add properties first.")
    else:
        with st.form("add_vacancy"):
            pid = st.selectbox("Property", props["id"].tolist(), format_func=prop_label)
            start = st.date_input("Vacancy start", value=date.today())
            end_known = st.checkbox("I know the vacancy end date", value=False)
            end = st.date_input("Vacancy end", value=date.today(), disabled=not end_known)
            notes = st.text_input("Notes (optional)")

            if st.form_submit_button("Save Vacancy"):
                end_str = end.isoformat() if end_known else ""
                exec_sql(
                    "INSERT INTO vacancies (property_id, start_date, end_date, notes) VALUES (?,?,?,?)",
                    (int(pid), start.isoformat(), end_str, notes.strip()),
                )
                st.success("Vacancy saved.")
                st.rerun()

    st.divider()
    st.subheader("Vacancy Log")
    if vacancies.empty:
        st.info("No vacancy periods logged yet.")
    else:
        v = vacancies.copy()
        prop_map = dict(zip(props["id"], props["nickname"])) if not props.empty else {}
        v["property_name"] = v["property_id"].map(lambda x: prop_map.get(x, str(x)))
        st.dataframe(v[["id", "property_name", "start_date", "end_date", "notes"]], use_container_width=True)


elif page == "Reports (Rent Roll + Tax) (Pro-Pro)":
    st.subheader("Rent Roll (Active Leases)")
    if leases.empty:
        st.info("No leases yet.")
    else:
        active = leases[leases["is_active"] == 1].copy()
        if active.empty:
            st.info("No active leases.")
        else:
            active["Property"] = active["property_id"].apply(lambda x: prop_label(int(x)))
            roll = active[["Property", "tenant_name", "start_date", "end_date", "rent_monthly", "security_deposit", "notes"]].copy()
            roll = roll.rename(
                columns={
                    "tenant_name": "Tenant",
                    "start_date": "Start",
                    "end_date": "End",
                    "rent_monthly": "Rent_Monthly",
                    "security_deposit": "Deposit",
                    "notes": "Notes",
                }
            )
            st.dataframe(roll, use_container_width=True)
            st.metric("Total Scheduled Monthly Rent", f"${float(roll['Rent_Monthly'].sum()):,.0f}")
            df_download_button(roll, "rent_roll.csv", "Download rent_roll.csv")

    st.divider()
    st.subheader("Year-End Tax Report (Schedule-E-Style)")
    st.caption("Groups ledger categories into common buckets. Confirm with your tax professional.")

    if txs.empty:
        st.info("No ledger transactions yet.")
    else:
        year = st.number_input("Tax year", min_value=2000, max_value=2100, value=date.today().year, step=1, key="tax_year")

        bucket_map = {
            "Rent": "Rents Received",
            "Mortgage": "Mortgage Interest / Debt Service (check breakdown)",
            "Taxes": "Taxes",
            "Insurance": "Insurance",
            "HOA": "HOA",
            "Utilities": "Utilities",
            "Management": "Management Fees",
            "Repairs": "Repairs",
            "CapEx": "Improvements (CapEx) (may be depreciated)",
            "Turnover": "Cleaning/Turnover",
            "Supplies": "Supplies",
            "Legal/Professional": "Legal & Professional",
            "Advertising": "Advertising",
            "Travel": "Travel",
            "Other": "Other",
        }

        df = txs.copy()
        df["tx_date"] = pd.to_datetime(df["tx_date"])
        df = df[df["tx_date"].dt.year == int(year)].copy()

        if df.empty:
            st.info("No transactions for that year.")
        else:
            df["signed"] = df.apply(lambda r: r["amount"] if r["tx_type"] == "INCOME" else -r["amount"], axis=1)
            df["Bucket"] = df["category"].map(lambda c: bucket_map.get(c, "Other"))
            df["Property"] = df["property_id"].map(lambda pid: prop_label(int(pid)))

            by_bucket = df.groupby("Bucket", as_index=False)["signed"].sum().rename(columns={"signed": "Net_Amount"}).sort_values("Net_Amount", ascending=False)
            st.write("### Totals by Tax Bucket")
            st.dataframe(by_bucket, use_container_width=True)

            by_prop_bucket = df.groupby(["Property", "Bucket"], as_index=False)["signed"].sum().rename(columns={"signed": "Net_Amount"})
            st.write("### By Property (Bucket breakdown)")
            st.dataframe(by_prop_bucket, use_container_width=True)

            income = float(df[df["tx_type"] == "INCOME"]["amount"].sum())
            expenses = float(df[df["tx_type"] == "EXPENSE"]["amount"].sum())
            net = income - expenses
            c1, c2, c3 = st.columns(3)
            c1.metric("Income", f"${income:,.0f}")
            c2.metric("Expenses", f"${expenses:,.0f}")
            c3.metric("Net", f"${net:,.0f}")

            df_download_button(by_bucket, f"tax_report_{year}_by_bucket.csv", "Download tax_report_by_bucket.csv")
            df_download_button(by_prop_bucket, f"tax_report_{year}_by_property.csv", "Download tax_report_by_property.csv")


elif page == "Net Worth Timeline (Upgraded)":
    st.subheader("Net Worth Timeline (Portfolio Value − Debt) — Upgraded")
    if props.empty:
        st.info("Add properties first.")
    else:
        months = st.slider("Projection length (months)", 12, 480, 240, 12)

        appreciation = st.number_input(
            "Annual appreciation rate (%)",
            min_value=-10.0,
            max_value=20.0,
            step=0.25,
            value=float(settings.get("appreciation_rate_annual", 3) or 3),
        )

        st.divider()
        st.write("### Debt Paydown Assumptions")
        extra_base = st.number_input("Extra principal you add monthly", min_value=0.0, step=50.0, value=500.0)
        include_rental_routing = st.checkbox("Add positive rental cashflow to extra payments", value=True)
        primary_first = st.checkbox("Primary-first payoff targeting", value=bool(int(settings.get("use_primary_first", 1))))
        reserve_lock = st.checkbox(
            "Enable reserve lock (stops extra if reserves below target)",
            value=bool(int(settings.get("include_reserve_lock", 1))),
        )
        reserve_months_target = st.number_input(
            "Reserve months target",
            min_value=0.0,
            step=1.0,
            value=float(settings.get("reserve_months_target", 3) or 3),
        )

        total_reserves = float(props["reserve_balance"].fillna(0).sum()) if not props.empty else 0.0
        required_reserves = compute_required_reserves(props, loans, rentals, float(reserve_months_target))
        gap = max(required_reserves - total_reserves, 0.0)

        r1, r2, r3 = st.columns(3)
        r1.metric("Reserves (sum)", f"${total_reserves:,.0f}")
        r2.metric("Required reserves", f"${required_reserves:,.0f}")
        r3.metric("Reserve gap", f"${gap:,.0f}")

        proj, meta = project_networth_over_time(
            props_df=props,
            loans_df=loans,
            rentals_df=rentals,
            appreciation_rate_annual_pct=float(appreciation),
            months=int(months),
            extra_base_monthly=float(extra_base),
            include_rental_routing=bool(include_rental_routing),
            primary_first=bool(primary_first),
            reserve_lock=bool(reserve_lock),
            reserve_months_target=float(reserve_months_target),
        )

        st.info(
            f"Extra base: ${meta['Extra_Base']:,.0f}/mo | "
            f"Routed rental CF: ${meta['Routed_Rental_CF']:,.0f}/mo | "
            f"Extra allowed after lock: ${meta['Extra_Allowed']:,.0f}/mo"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Starting Net Worth (approx)", f"${proj['Net_Worth'].iloc[0]:,.0f}")
        c2.metric("Net Worth at end (approx)", f"${proj['Net_Worth'].iloc[-1]:,.0f}")
        c3.metric("Debt at end (approx)", f"${proj['Debt'].iloc[-1]:,.0f}")

        chart_df = proj.copy()
        chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        st.write("### Net Worth Over Time")
        st.line_chart(chart_df.set_index("Date")[["Net_Worth"]])

        st.write("### Value vs Debt Over Time")
        st.line_chart(chart_df.set_index("Date")[["Portfolio_Value", "Debt"]])

        st.write("### Table (first 120 months)")
        view = proj.copy()
        view["Date"] = view["Date"].astype(str)
        st.dataframe(view.head(120), use_container_width=True)
        df_download_button(proj, "net_worth_timeline.csv", "Download net_worth_timeline.csv")


elif page == "Backup & Restore (Pro-Pro)":
    st.subheader("Backup & Restore (Your Account Only)")
    st.caption("Backup includes your private user DB + your receipts folder only.")

    uid = st.session_state.user_id
    user_db = user_db_path(uid)
    uploads_dir = user_upload_dir(uid)

    st.write("### Backup")
    backup_bytes = build_backup_zip_bytes(user_db, uploads_dir)
    st.download_button(
        "Download backup.zip (your data)",
        data=backup_bytes,
        file_name=f"va_empire_backup_user{uid}_{date.today().isoformat()}.zip",
        mime="application/zip",
    )

    st.divider()
    st.write("### Restore")
    st.warning("Restore overwrites your current data for this account.")
    confirm = st.checkbox("I understand and want to restore from a backup ZIP")
    up = st.file_uploader("Upload backup.zip", type=["zip"])
    if confirm and up is not None:
        if st.button("Restore Now"):
            try:
                safe_restore_from_zip(up.getvalue(), user_db, uploads_dir)
                st.success("Restore completed. Refreshing…")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")


elif page == "Exports":
    st.subheader("Export Data to CSV (Your Account)")
    df_download_button(props, "properties.csv", "Download properties.csv")
    df_download_button(loans, "loans.csv", "Download loans.csv")
    df_download_button(rentals, "rentals.csv", "Download rentals.csv")
    df_download_button(tenants, "tenants.csv", "Download tenants.csv")
    df_download_button(leases, "leases.csv", "Download leases.csv")
    df_download_button(txs, "transactions.csv", "Download transactions.csv")
    df_download_button(vacancies, "vacancies.csv", "Download vacancies.csv")
    df_download_button(attachments, "attachments.csv", "Download attachments.csv")


elif page == "Settings":
    st.subheader("Settings")

    county_loan_limit = st.number_input(
        "County loan limit (planner input)",
        min_value=0.0,
        step=1000.0,
        value=float(settings.get("county_loan_limit", 0) or 0),
    )
    reserve_months = st.number_input(
        "Reserve months target",
        min_value=0.0,
        step=1.0,
        value=float(settings.get("reserve_months_target", 3) or 3),
    )
    use_primary_first = st.checkbox(
        "Default to Primary-first payoff (used in timeline)",
        value=bool(int(settings.get("use_primary_first", 1))),
    )
    appreciation_rate = st.number_input(
        "Default annual appreciation rate (%)",
        min_value=-10.0,
        max_value=20.0,
        step=0.25,
        value=float(settings.get("appreciation_rate_annual", 3) or 3),
    )
    include_reserve_lock = st.checkbox(
        "Enable reserve safety lock by default (used in timeline)",
        value=bool(int(settings.get("include_reserve_lock", 1))),
    )

    if st.button("Save Settings"):
        exec_sql(
            """
            UPDATE settings
            SET county_loan_limit=?,
                reserve_months_target=?,
                use_primary_first=?,
                appreciation_rate_annual=?,
                include_reserve_lock=?
            WHERE id=1
            """,
            (
                float(county_loan_limit),
                float(reserve_months),
                1 if use_primary_first else 0,
                float(appreciation_rate),
                1 if include_reserve_lock else 0,
            ),
        )
        st.success("Saved.")
        st.rerun()
