"""
School System backend
---------------------
FastAPI + PostgreSQL backend for student records, fee sheets, roles, audit logs,
and daybook entries.

Install:
    pip install fastapi uvicorn pydantic psycopg2-binary

Configuration (Environment Variables):
    DATABASE_URL=postgresql://user:password@localhost:5432/school_system
    -- or individual settings --
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB=school_system
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=postgres

    SCHOOL_SUPERADMIN_USERNAME=superadmin
    SCHOOL_SUPERADMIN_PASSWORD=change-this-password
    SCHOOL_TOKEN_SECRET=optional-random-hex-secret
    SCHOOL_TOKEN_TTL_SECONDS=28800

Run:
    uvicorn backend:app --reload

Security notes:
- Passwords are PBKDF2-HMAC-SHA256 hashed (never stored as plaintext).
- Login returns a signed bearer token; protected APIs derive the acting user/role
  from that token, not from caller-supplied role fields.
- Configure a strong initial super-admin password using environment variables.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()



APP_NAME = "School System"

# PostgreSQL Configuration
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "localhost"))
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")))
POSTGRES_DB = os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "school_system"))
POSTGRES_USER = os.getenv("POSTGRES_USER", os.getenv("PGUSER", "postgres"))
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", "postgres"))

TOKEN_SECRET = os.getenv("SCHOOL_TOKEN_SECRET", "")
if not TOKEN_SECRET:
    # For development convenience. Set SCHOOL_TOKEN_SECRET to a stable,
    # high-entropy secret in production or tokens will expire on restart.
    TOKEN_SECRET = secrets.token_hex(32)

TOKEN_TTL_SECONDS = int(os.getenv("SCHOOL_TOKEN_TTL_SECONDS", "28800"))
FEE_TYPES = {
    "tuition_fee": "Tuition Fee",
    "hostel_fee": "Hostel Fee",
    "bus_fee": "Bus Fee",
    "exam_fee": "Exam Fee",
    "other_fee": "Other Fee",
}
ADMIN_ROLES = {"admin", "super_admin"}
SUPERADMIN_ROLES = {"super_admin"}

app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
_db_pool: Optional[pool.ThreadedConnectionPool] = None


class DBConnectionWrapper:
    """
    Lightweight wrapper around a psycopg2 connection to provide
    a convenient .execute() method returning a RealDictCursor.
    """
    def __init__(self, raw_conn: Any):
        self._conn = raw_conn

    @property
    def raw_conn(self) -> Any:
        return self._conn

    def execute(self, query: str, params: Any = None) -> Any:
        cur = self._conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        if "cursor_factory" not in kwargs:
            kwargs["cursor_factory"] = RealDictCursor
        return self._conn.cursor(*args, **kwargs)


def _ensure_database_exists() -> None:
    """
    Helper to check if the target database exists on the PostgreSQL server,
    and create it if it does not already exist.
    """
    target_db = POSTGRES_DB
    connect_kwargs: dict[str, Any] = {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "dbname": "postgres",
    }

    if DATABASE_URL:
        try:
            parsed = urlparse(DATABASE_URL)
            if parsed.path and len(parsed.path) > 1:
                target_db = parsed.path.lstrip("/")
            connect_kwargs = {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 5432,
                "user": parsed.username or "postgres",
                "password": parsed.password or "postgres",
                "dbname": "postgres",
            }
        except Exception:
            pass

    try:
        tmp_conn = psycopg2.connect(**connect_kwargs)
        tmp_conn.autocommit = True
        with tmp_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if not cur.fetchone():
                # Database identifiers must be validated
                sanitized_db = target_db.replace('"', '""')
                cur.execute(f'CREATE DATABASE "{sanitized_db}"')
        tmp_conn.close()
    except Exception:
        # If user lacks permission to connect to 'postgres' or create db,
        # proceed and let the main connection attempt connect directly.
        pass


def get_pool() -> pool.ThreadedConnectionPool:
    global _db_pool
    if _db_pool is None or _db_pool.closed:
        _ensure_database_exists()
        if DATABASE_URL:
            _db_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=DATABASE_URL,
            )
        else:
            _db_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
            )
    return _db_pool


@contextmanager
def db_connect() -> Generator[DBConnectionWrapper, None, None]:
    """
    Context manager yielding a pooled PostgreSQL connection with transaction handling.
    Commits on successful completion, rolls back on exceptions, and returns
    the connection back to the connection pool.
    """
    pool_instance = get_pool()
    conn = pool_instance.getconn()
    wrapper = DBConnectionWrapper(conn)
    try:
        with conn:
            yield wrapper
    finally:
        pool_instance.putconn(conn)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def init_db() -> None:
    """Initialize PostgreSQL database schema and seed the initial super-admin user."""
    with db_connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            roll_no VARCHAR(100) NOT NULL UNIQUE,
            name VARCHAR(255) NOT NULL,
            father_name VARCHAR(255),
            class VARCHAR(100),
            is_verified INTEGER NOT NULL DEFAULT 0,
            verified_by VARCHAR(100) DEFAULT NULL,
            created_by VARCHAR(100) NOT NULL,
            created_at VARCHAR(100) NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fees (
            id SERIAL PRIMARY KEY,
            roll_no VARCHAR(100) NOT NULL,
            sheet_type VARCHAR(100) NOT NULL DEFAULT 'Tuition Fee',
            fixed_installment DOUBLE PRECISION NOT NULL,
            pending_installment DOUBLE PRECISION NOT NULL,
            is_verified INTEGER NOT NULL DEFAULT 0,
            verified_by VARCHAR(100) DEFAULT NULL,
            created_by VARCHAR(100) NOT NULL,
            created_at VARCHAR(100) NOT NULL,
            UNIQUE(roll_no, sheet_type),
            FOREIGN KEY(roll_no) REFERENCES students(roll_no) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            created_at VARCHAR(100) NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100),
            role VARCHAR(50),
            action VARCHAR(255) NOT NULL,
            details TEXT,
            timestamp VARCHAR(100) NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daybook (
            id SERIAL PRIMARY KEY,
            sheet_type VARCHAR(100) NOT NULL DEFAULT 'Tuition Fee',
            code VARCHAR(100),
            roll_no VARCHAR(100),
            student_name VARCHAR(255),
            father_name VARCHAR(255),
            class VARCHAR(100),
            pending_fee_month VARCHAR(100),
            fine DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            date VARCHAR(50) NOT NULL,
            remarks TEXT,
            category VARCHAR(100),
            credit DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            debit DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            balance DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            voucher_no VARCHAR(100),
            challan_no VARCHAR(100),
            is_verified INTEGER NOT NULL DEFAULT 0,
            verified_by VARCHAR(100),
            created_by VARCHAR(100) NOT NULL,
            amended_by VARCHAR(100),
            amended_at VARCHAR(100)
        );

        CREATE INDEX IF NOT EXISTS idx_students_name ON students(name);
        CREATE INDEX IF NOT EXISTS idx_fees_roll_no ON fees(roll_no);
        CREATE INDEX IF NOT EXISTS idx_daybook_date ON daybook(date);
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON audit_logs(timestamp);
        """)

        # Create the first super-admin only when the roles table is empty
        username = os.getenv("SCHOOL_SUPERADMIN_USERNAME", "uditya")
        password = os.getenv("SCHOOL_SUPERADMIN_PASSWORD", "uditya123")
        cur = conn.execute("SELECT COUNT(*) AS cnt FROM roles")
        count = cur.fetchone()["cnt"]
        if count == 0 and username and password:
            conn.execute(
                """INSERT INTO roles(username, password_hash, full_name, role, created_at, is_active)
                   VALUES(%s, %s, %s, %s, %s, 1)""",
                (username.strip(), hash_password(password), "System Super Admin",
                 "super_admin", now_iso()),
            )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def check_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def make_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    import base64
    body = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    signature = hmac.new(TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def decode_token(token: str) -> dict[str, Any]:
    import base64
    try:
        body, signature = token.split(".", 1)
        expected = hmac.new(TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        payload = json.loads(raw)
        if int(payload["exp"]) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired bearer token")


def current_user(authorization: Optional[str] = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    payload = decode_token(authorization.split(" ", 1)[1].strip())
    with db_connect() as conn:
        row = conn.execute(
            "SELECT username, role, is_active FROM roles WHERE username=%s",
            (payload.get("sub"),),
        ).fetchone()
    if not row or not row["is_active"] or row["role"] != payload.get("role"):
        raise HTTPException(status_code=401, detail="User is inactive or token is no longer valid")
    return {"username": row["username"], "role": row["role"]}


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"].lower() not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin or SuperAdmin permission required")
    return user


def require_superadmin(user: dict = Depends(current_user)) -> dict:
    if user["role"].lower() not in SUPERADMIN_ROLES:
        raise HTTPException(status_code=403, detail="SuperAdmin permission required")
    return user


def log_action(conn: Any, user: dict, action: str, details: Any = None) -> None:
    conn.execute(
        "INSERT INTO audit_logs(username, role, action, details, timestamp) VALUES(%s, %s, %s, %s, %s)",
        (user.get("username"), user.get("role"), action,
         json.dumps(details, ensure_ascii=False, default=str) if details is not None else None,
         now_iso()),
    )


def normalize_sheet_type(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    if key not in FEE_TYPES:
        raise HTTPException(status_code=422, detail=f"sheet_type must be one of: {', '.join(FEE_TYPES)}")
    return FEE_TYPES[key]


def row_dict(row: Optional[Any]) -> Optional[dict]:
    return dict(row) if row is not None else None


class LoginIn(BaseModel):
    username: str
    password: str


class StudentCreate(BaseModel):
    roll_no: str = Field(min_length=1)
    name: str = Field(min_length=1)
    father_name: Optional[str] = None
    class_name: Optional[str] = Field(default=None, alias="class")
    class Config:
        populate_by_name = True


class StudentEdit(BaseModel):
    name: Optional[str] = None
    father_name: Optional[str] = None
    class_name: Optional[str] = Field(default=None, alias="class")
    class Config:
        populate_by_name = True


class FeesCreate(BaseModel):
    roll_no: str
    sheet_type: str = "Tuition Fee"
    fixed_installment: float = Field(ge=0)
    pending_installment: Optional[float] = Field(default=None, ge=0)


class FeesEdit(BaseModel):
    fixed_installment: Optional[float] = Field(default=None, ge=0)
    pending_installment: Optional[float] = Field(default=None, ge=0)


class RoleCreate(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    is_active: bool = True


class RoleEdit(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class PasswordChange(BaseModel):
    new_password: str = Field(min_length=8)


class LogCreate(BaseModel):
    action: str
    details: Optional[str] = None


class DaybookCreate(BaseModel):
    sheet_type: str = "Tuition Fee"
    code: Optional[str] = None
    roll_no: Optional[str] = None
    student_name: Optional[str] = None
    father_name: Optional[str] = None
    class_name: Optional[str] = Field(default=None, alias="class")
    pending_fee_month: Optional[str] = None
    fine: float = 0.0
    date: str
    remarks: Optional[str] = None
    category: Optional[str] = None
    credit: float = 0.0
    debit: float = 0.0
    balance: float = 0.0
    voucher_no: Optional[str] = None
    challan_no: Optional[str] = None
    class Config:
        populate_by_name = True


class DaybookEdit(BaseModel):
    sheet_type: Optional[str] = None
    code: Optional[str] = None
    roll_no: Optional[str] = None
    student_name: Optional[str] = None
    father_name: Optional[str] = None
    class_name: Optional[str] = Field(default=None, alias="class")
    pending_fee_month: Optional[str] = None
    fine: Optional[float] = None
    date: Optional[str] = None
    remarks: Optional[str] = None
    category: Optional[str] = None
    credit: Optional[float] = None
    debit: Optional[float] = None
    balance: Optional[float] = None
    voucher_no: Optional[str] = None
    challan_no: Optional[str] = None
    class Config:
        populate_by_name = True


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.on_event("shutdown")
def shutdown() -> None:
    global _db_pool
    if _db_pool is not None and not _db_pool.closed:
        _db_pool.closeall()


@app.get("/")
def root():
    return {"name": APP_NAME, "status": "ok", "docs": "/docs"}


@app.post("/auth/login")
def login(data: LoginIn):
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM roles WHERE username=%s", (data.username,)).fetchone()
    if not row or not row["is_active"] or not check_password(data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user = {"username": row["username"], "role": row["role"]}
    with db_connect() as conn:
        log_action(conn, user, "login")
    return {"access_token": make_token(user["username"], user["role"]),
            "token_type": "bearer", "expires_in": TOKEN_TTL_SECONDS,
            "username": user["username"], "role": user["role"]}


# STUDENTS
@app.post("/students", status_code=201)
def create_student(data: StudentCreate, user: dict = Depends(current_user)):
    with db_connect() as conn:
        try:
            cur = conn.execute(
                """INSERT INTO students(roll_no, name, father_name, class, is_verified, verified_by, created_by, created_at)
                   VALUES(%s, %s, %s, %s, 0, NULL, %s, %s)
                   RETURNING *""",
                (data.roll_no.strip(), data.name.strip(), data.father_name,
                 data.class_name, user["username"], now_iso()),
            )
            row = cur.fetchone()
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=409, detail="roll_no already exists")
        log_action(conn, user, "create_student", {"roll_no": data.roll_no})
        return row_dict(row)


@app.post("/students/{roll_no}/verify")
def verify_student(roll_no: str, user: dict = Depends(require_admin)):
    with db_connect() as conn:
        cur = conn.execute("UPDATE students SET is_verified=1, verified_by=%s WHERE roll_no=%s RETURNING *",
                           (user["username"], roll_no))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Student not found")
        log_action(conn, user, "verify_student", {"roll_no": roll_no})
        return row_dict(row)


@app.post("/students/{roll_no}/unverify")
def unverify_student(roll_no: str, user: dict = Depends(require_admin)):
    with db_connect() as conn:
        cur = conn.execute("UPDATE students SET is_verified=0, verified_by=NULL WHERE roll_no=%s RETURNING *", (roll_no,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Student not found")
        log_action(conn, user, "unverify_student", {"roll_no": roll_no})
        return row_dict(row)


@app.put("/students/{roll_no}")
def edit_student(roll_no: str, data: StudentEdit, user: dict = Depends(require_admin)):
    fields = data.model_dump(by_alias=False, exclude_unset=True)
    mapping = {"name": "name", "father_name": "father_name", "class_name": "class"}
    updates = [(mapping[k], v) for k, v in fields.items()]
    if not updates:
        raise HTTPException(status_code=400, detail="No editable fields supplied")
    with db_connect() as conn:
        check = conn.execute("SELECT 1 FROM students WHERE roll_no=%s", (roll_no,)).fetchone()
        if not check:
            raise HTTPException(status_code=404, detail="Student not found")
        sql = "UPDATE students SET " + ", ".join(f'"{k}"=%s' for k, _ in updates) + " WHERE roll_no=%s RETURNING *"
        cur = conn.execute(sql, [v for _, v in updates] + [roll_no])
        log_action(conn, user, "edit_student", {"roll_no": roll_no, "fields": [k for k, _ in updates]})
        return row_dict(cur.fetchone())


@app.delete("/students/{roll_no}")
def delete_student(roll_no: str, user: dict = Depends(require_admin)):
    with db_connect() as conn:
        cur = conn.execute("DELETE FROM students WHERE roll_no=%s", (roll_no,))
        if not cur.rowcount:
            raise HTTPException(status_code=404, detail="Student not found")
        log_action(conn, user, "delete_student", {"roll_no": roll_no})
    return {"message": "Student deleted", "roll_no": roll_no}


@app.get("/students/{roll_no}")
def get_student(roll_no: str, user: dict = Depends(current_user)):
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM students WHERE roll_no=%s", (roll_no,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    return row_dict(row)


@app.get("/students")
def get_students(name: str = Query(..., min_length=1), user: dict = Depends(current_user)):
    with db_connect() as conn:
        rows = conn.execute("SELECT * FROM students WHERE name ILIKE %s ORDER BY name",
                            (f"%{name}%",)).fetchall()
    return [dict(r) for r in rows]


# FEES
@app.post("/fees", status_code=201)
def add_fees(data: FeesCreate, user: dict = Depends(require_admin)):
    sheet_type = normalize_sheet_type(data.sheet_type)
    pending = data.pending_installment if data.pending_installment is not None else data.fixed_installment
    with db_connect() as conn:
        if not conn.execute("SELECT 1 FROM students WHERE roll_no=%s", (data.roll_no,)).fetchone():
            raise HTTPException(status_code=404, detail="roll_no does not exist in Student table")
        try:
            cur = conn.execute(
                """INSERT INTO fees(roll_no, sheet_type, fixed_installment, pending_installment,
                   is_verified, verified_by, created_by, created_at)
                   VALUES(%s, %s, %s, %s, 0, NULL, %s, %s)
                   RETURNING *""",
                (data.roll_no, sheet_type, data.fixed_installment, pending,
                 user["username"], now_iso()),
            )
            row = cur.fetchone()
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=409, detail="Fee sheet already exists for this roll_no and sheet_type")
        log_action(conn, user, "add_fees", {"roll_no": data.roll_no, "sheet_type": sheet_type})
        return row_dict(row)


@app.put("/fees/{roll_no}/{sheet_type}")
def edit_fees(roll_no: str, sheet_type: str, data: FeesEdit, user: dict = Depends(require_admin)):
    normalized = normalize_sheet_type(sheet_type)
    fields = data.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fee fields supplied")
    with db_connect() as conn:
        cur = conn.execute(
            "UPDATE fees SET " + ", ".join(f'"{k}"=%s' for k in fields) +
            " WHERE roll_no=%s AND sheet_type=%s RETURNING *",
            list(fields.values()) + [roll_no, normalized],
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Fee sheet not found")
        log_action(conn, user, "edit_fees", {"roll_no": roll_no, "sheet_type": normalized})
        return row_dict(row)


@app.post("/fees/{roll_no}/{sheet_type}/verify")
def verify_fees(roll_no: str, sheet_type: str, user: dict = Depends(require_admin)):
    normalized = normalize_sheet_type(sheet_type)
    with db_connect() as conn:
        cur = conn.execute("UPDATE fees SET is_verified=1, verified_by=%s WHERE roll_no=%s AND sheet_type=%s RETURNING *",
                           (user["username"], roll_no, normalized))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Fee sheet not found")
        log_action(conn, user, "verify_fees", {"roll_no": roll_no, "sheet_type": normalized})
        return row_dict(row)


@app.post("/fees/{roll_no}/{sheet_type}/unverify")
def unverify_fees(roll_no: str, sheet_type: str, user: dict = Depends(require_admin)):
    normalized = normalize_sheet_type(sheet_type)
    with db_connect() as conn:
        cur = conn.execute("UPDATE fees SET is_verified=0, verified_by=NULL WHERE roll_no=%s AND sheet_type=%s RETURNING *",
                           (roll_no, normalized))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Fee sheet not found")
        log_action(conn, user, "unverify_fees", {"roll_no": roll_no, "sheet_type": normalized})
        return row_dict(row)


@app.get("/fees/{roll_no}")
def get_fees(roll_no: str, user: dict = Depends(current_user)):
    with db_connect() as conn:
        rows = conn.execute("SELECT * FROM fees WHERE roll_no=%s ORDER BY sheet_type", (roll_no,)).fetchall()
    return [dict(r) for r in rows]


# ROLES (role administration restricted to super-admin)
@app.post("/roles", status_code=201)
def create_role(data: RoleCreate, user: dict = Depends(require_superadmin)):
    with db_connect() as conn:
        try:
            cur = conn.execute(
                """INSERT INTO roles(username, password_hash, full_name, role, created_at, is_active)
                   VALUES(%s, %s, %s, %s, %s, %s)
                   RETURNING id, username, full_name, role, created_at, is_active""",
                (data.username.strip(), hash_password(data.password), data.full_name.strip(),
                 data.role.strip().lower(), now_iso(), int(data.is_active)),
            )
            row = cur.fetchone()
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=409, detail="Username already exists")
        log_action(conn, user, "create_role", {"username": data.username, "role": data.role})
        return row_dict(row)


@app.put("/roles/{username}")
def edit_role(username: str, data: RoleEdit, user: dict = Depends(require_superadmin)):
    fields = data.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields supplied")
    if "role" in fields and fields["role"]:
        fields["role"] = fields["role"].strip().lower()
    if "is_active" in fields and fields["is_active"] is not None:
        fields["is_active"] = int(fields["is_active"])
    with db_connect() as conn:
        cur = conn.execute("UPDATE roles SET " + ", ".join(f'"{k}"=%s' for k in fields) +
                           " WHERE username=%s RETURNING id, username, full_name, role, created_at, is_active",
                           list(fields.values()) + [username])
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Role/user not found")
        log_action(conn, user, "edit_role", {"username": username, "fields": list(fields)})
        return row_dict(row)


@app.delete("/roles/{username}")
def delete_role(username: str, user: dict = Depends(require_superadmin)):
    if username == user["username"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    with db_connect() as conn:
        cur = conn.execute("DELETE FROM roles WHERE username=%s", (username,))
        if not cur.rowcount:
            raise HTTPException(status_code=404, detail="Role/user not found")
        log_action(conn, user, "delete_role", {"username": username})
    return {"message": "Role deleted", "username": username}


@app.post("/roles/{username}/change-password")
def change_password(username: str, data: PasswordChange, user: dict = Depends(current_user)):
    if user["role"].lower() not in SUPERADMIN_ROLES and user["username"] != username:
        raise HTTPException(status_code=403, detail="You can change only your own password")
    with db_connect() as conn:
        cur = conn.execute("UPDATE roles SET password_hash=%s WHERE username=%s",
                           (hash_password(data.new_password), username))
        if not cur.rowcount:
            raise HTTPException(status_code=404, detail="Role/user not found")
        log_action(conn, user, "change_password", {"target_username": username})
    return {"message": "Password changed. Please log in again."}


@app.get("/roles")
def get_roles(user: dict = Depends(require_superadmin)):
    with db_connect() as conn:
        rows = conn.execute("SELECT id, username, full_name, role, created_at, is_active FROM roles ORDER BY id").fetchall()
    return [dict(r) for r in rows]


@app.get("/roles/{username}")
def get_role(username: str, user: dict = Depends(require_superadmin)):
    with db_connect() as conn:
        row = conn.execute("SELECT id, username, full_name, role, created_at, is_active FROM roles WHERE username=%s",
                           (username,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Role/user not found")
    return row_dict(row)


# AUDIT LOGS
@app.post("/logs", status_code=201)
def new_log(data: LogCreate, user: dict = Depends(current_user)):
    with db_connect() as conn:
        log_action(conn, user, data.action, data.details)
        return {"message": "Log created"}


@app.get("/logs")
def get_logs(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
             user: dict = Depends(require_admin)):
    with db_connect() as conn:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT %s OFFSET %s",
                            (limit, offset)).fetchall()
    return [dict(r) for r in rows]


# DAYBOOK
DAYBOOK_COLUMNS = {
    "sheet_type": "sheet_type", "code": "code", "roll_no": "roll_no",
    "student_name": "student_name", "father_name": "father_name",
    "class_name": "class", "pending_fee_month": "pending_fee_month",
    "fine": "fine", "date": "date", "remarks": "remarks", "category": "category",
    "credit": "credit", "debit": "debit", "balance": "balance",
    "voucher_no": "voucher_no", "challan_no": "challan_no",
}


def daybook_payload(data: BaseModel) -> dict:
    raw = data.model_dump(by_alias=False, exclude_unset=True)
    return {DAYBOOK_COLUMNS.get(k, k): v for k, v in raw.items()}


def adjust_pending_fees(
    conn: Any,
    roll_no: Optional[str],
    sheet_type: Optional[str],
    delta: float,
    user: dict,
    daybook_id: int,
) -> Optional[float]:
    """
    Adjusts the pending_installment for the student's fee record.
    A positive delta deducts from pending_installment (e.g. debit=3000 -> pending decreases by 3000).
    A negative delta restores to pending_installment (e.g. entry deletion).
    """
    if not roll_no or not delta:
        return None

    raw_sheet = sheet_type or "Tuition Fee"
    try:
        norm_sheet = normalize_sheet_type(raw_sheet)
    except HTTPException:
        norm_sheet = raw_sheet

    fee = conn.execute(
        """SELECT id, fixed_installment, pending_installment, sheet_type 
           FROM fees 
           WHERE roll_no=%s AND (sheet_type=%s OR LOWER(sheet_type)=LOWER(%s))""",
        (roll_no, norm_sheet, norm_sheet),
    ).fetchone()

    if not fee:
        return None

    old_pending = float(fee["pending_installment"])
    fixed = float(fee["fixed_installment"])
    new_pending = max(0.0, min(fixed, round(old_pending - delta, 2)))

    conn.execute(
        "UPDATE fees SET pending_installment=%s WHERE id=%s",
        (new_pending, fee["id"]),
    )
    log_action(conn, user, "adjust_pending_fees", {
        "roll_no": roll_no,
        "sheet_type": fee["sheet_type"],
        "delta": delta,
        "old_pending": old_pending,
        "new_pending": new_pending,
        "daybook_id": daybook_id,
    })
    return new_pending


@app.post("/daybook", status_code=201)
def create_entry(data: DaybookCreate, user: dict = Depends(current_user)):
    payload = daybook_payload(data)
    if payload.get("sheet_type"):
        try:
            payload["sheet_type"] = normalize_sheet_type(payload["sheet_type"])
        except HTTPException:
            pass
    payload["created_by"] = user["username"]
    payload["is_verified"] = 0
    payload["verified_by"] = None
    columns = list(payload)
    with db_connect() as conn:
        quoted_cols = [f'"{c}"' for c in columns]
        cur = conn.execute(
            f"INSERT INTO daybook({','.join(quoted_cols)}) VALUES({','.join('%s' for _ in columns)}) RETURNING id",
            [payload[c] for c in columns],
        )
        entry_row = cur.fetchone()
        entry_id = entry_row["id"]
        log_action(conn, user, "create_daybook_entry", {"id": entry_id})

        debit = float(payload.get("debit") or 0.0)
        credit = float(payload.get("credit") or 0.0)
        pay_amt = debit if debit > 0 else (credit if credit > 0 else 0.0)

        adjust_pending_fees(
            conn=conn,
            roll_no=payload.get("roll_no"),
            sheet_type=payload.get("sheet_type"),
            delta=pay_amt,
            user=user,
            daybook_id=entry_id,
        )

        row = conn.execute("SELECT * FROM daybook WHERE id=%s", (entry_id,)).fetchone()
        return row_dict(row)


@app.post("/daybook/{entry_id}/verify")
def verified_entry(entry_id: int, user: dict = Depends(require_admin)):
    with db_connect() as conn:
        cur = conn.execute("UPDATE daybook SET is_verified=1, verified_by=%s WHERE id=%s RETURNING *",
                           (user["username"], entry_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Daybook entry not found")
        log_action(conn, user, "verify_daybook_entry", {"id": entry_id})
        return row_dict(row)


@app.post("/daybook/{entry_id}/unverify")
def unverified_entry(entry_id: int, user: dict = Depends(require_admin)):
    with db_connect() as conn:
        cur = conn.execute("UPDATE daybook SET is_verified=0, verified_by=NULL WHERE id=%s RETURNING *", (entry_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Daybook entry not found")
        log_action(conn, user, "unverify_daybook_entry", {"id": entry_id})
        return row_dict(row)


@app.put("/daybook/{entry_id}/amend")
def amended_entry(entry_id: int, data: DaybookEdit, user: dict = Depends(require_admin)):
    payload = daybook_payload(data)
    if not payload:
        raise HTTPException(status_code=400, detail="No amendment fields supplied")
    payload["amended_by"] = user["username"]
    payload["amended_at"] = now_iso()
    columns = list(payload)
    with db_connect() as conn:
        old_row = conn.execute("SELECT * FROM daybook WHERE id=%s", (entry_id,)).fetchone()
        if not old_row:
            raise HTTPException(status_code=404, detail="Daybook entry not found")
        cur = conn.execute("UPDATE daybook SET " + ", ".join(f'"{c}"=%s' for c in columns) +
                           " WHERE id=%s RETURNING *", [payload[c] for c in columns] + [entry_id])
        updated_row = cur.fetchone()
        log_action(conn, user, "amend_daybook_entry", {"id": entry_id, "fields": columns})

        if "debit" in payload or "credit" in payload:
            old_debit = float(old_row["debit"] or 0.0)
            old_credit = float(old_row["credit"] or 0.0)
            old_amt = old_debit if old_debit > 0 else old_credit

            new_debit = float(payload["debit"]) if "debit" in payload else old_debit
            new_credit = float(payload["credit"]) if "credit" in payload else old_credit
            new_amt = new_debit if new_debit > 0 else new_credit

            delta = round(new_amt - old_amt, 2)
            if delta != 0:
                adjust_pending_fees(
                    conn=conn,
                    roll_no=payload.get("roll_no") or old_row["roll_no"],
                    sheet_type=payload.get("sheet_type") or old_row["sheet_type"],
                    delta=delta,
                    user=user,
                    daybook_id=entry_id,
                )

        return row_dict(updated_row)


@app.put("/daybook/{entry_id}")
def edit_entry(entry_id: int, data: DaybookEdit, user: dict = Depends(require_admin)):
    payload = daybook_payload(data)
    if not payload:
        raise HTTPException(status_code=400, detail="No correction fields supplied")
    columns = list(payload)
    with db_connect() as conn:
        old_row = conn.execute("SELECT * FROM daybook WHERE id=%s", (entry_id,)).fetchone()
        if not old_row:
            raise HTTPException(status_code=404, detail="Daybook entry not found")
        cur = conn.execute("UPDATE daybook SET " + ", ".join(f'"{c}"=%s' for c in columns) +
                           " WHERE id=%s RETURNING *", [payload[c] for c in columns] + [entry_id])
        updated_row = cur.fetchone()
        log_action(conn, user, "edit_daybook_entry", {"id": entry_id, "fields": columns})

        if "debit" in payload or "credit" in payload:
            old_debit = float(old_row["debit"] or 0.0)
            old_credit = float(old_row["credit"] or 0.0)
            old_amt = old_debit if old_debit > 0 else old_credit

            new_debit = float(payload["debit"]) if "debit" in payload else old_debit
            new_credit = float(payload["credit"]) if "credit" in payload else old_credit
            new_amt = new_debit if new_debit > 0 else new_credit

            delta = round(new_amt - old_amt, 2)
            if delta != 0:
                adjust_pending_fees(
                    conn=conn,
                    roll_no=payload.get("roll_no") or old_row["roll_no"],
                    sheet_type=payload.get("sheet_type") or old_row["sheet_type"],
                    delta=delta,
                    user=user,
                    daybook_id=entry_id,
                )

        return row_dict(updated_row)


@app.delete("/daybook/{entry_id}")
def delete_entry(entry_id: int, user: dict = Depends(require_superadmin)):
    with db_connect() as conn:
        old_row = conn.execute("SELECT * FROM daybook WHERE id=%s", (entry_id,)).fetchone()
        if not old_row:
            raise HTTPException(status_code=404, detail="Daybook entry not found")
        cur = conn.execute("DELETE FROM daybook WHERE id=%s", (entry_id,))
        log_action(conn, user, "delete_daybook_entry", {"id": entry_id})

        old_debit = float(old_row["debit"] or 0.0)
        old_credit = float(old_row["credit"] or 0.0)
        old_amt = old_debit if old_debit > 0 else old_credit

        if old_amt > 0:
            adjust_pending_fees(
                conn=conn,
                roll_no=old_row["roll_no"],
                sheet_type=old_row["sheet_type"],
                delta=-old_amt,
                user=user,
                daybook_id=entry_id,
            )
    return {"message": "Daybook entry deleted", "id": entry_id}


@app.get("/daybook")
def get_daybook(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                roll_no: Optional[str] = None, user: dict = Depends(current_user)):
    sql = "SELECT * FROM daybook"
    params: list[Any] = []
    if roll_no:
        sql += " WHERE roll_no=%s"
        params.append(roll_no)
    sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    with db_connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/daybook/{entry_id}")
def get_daybook_entry(entry_id: int, user: dict = Depends(current_user)):
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM daybook WHERE id=%s", (entry_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Daybook entry not found")
    return row_dict(row)
