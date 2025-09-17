#!/usr/bin/env python3
# nl_sqlite_agent_II.py
# Agente NL→SQL seguro para SQLite (solo lectura) con PK/FK, relaciones, vistas, \schema y gráficos

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import math
from datetime import datetime
from dateutil import parser as dtparser  # pip install python-dateutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib; print(matplotlib.get_backend())

# ---------------- Config global ----------------
CHARTS_MODE = "auto"  # "auto" | "off"

# Cargar .env: primero desde el cwd y, si no aparece, el .env junto al script
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(f"No se encontró OPENAI_API_KEY. Revisa tu .env en: {dotenv_path}")

# Cliente OpenAI (se crea una vez)
from openai import OpenAI
client = OpenAI(api_key=api_key)

import re, sqlite3, json, argparse, textwrap
from tabulate import tabulate

# --- Configuración del LLM (OpenAI) ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # ajusta si lo deseas

# --- Reglas de seguridad sobre SQL generado ---
SQL_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|TRIGGER)\b",
    re.IGNORECASE,
)
SQL_MULTISTMT = re.compile(r";")  # prohibimos múltiples sentencias
SQL_REQUIRE_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
SQL_REQUIRE_LIMIT = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)

SYSTEM_INSTRUCTIONS = """
Eres un traductor NL→SQL para SQLite. Reglas:
- Devuelve una consulta SQL válida, SIN texto adicional.
- SOLO consultas SELECT (lectura). Prohibido: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, ATTACH, DETACH, PRAGMA, VACUUM, TRIGGER.
- NUNCA incluyas ';' al final.
- Incluye siempre LIMIT razonable (por defecto 500) si el usuario no especifica límite.
- Usa exclusivamente tablas/columnas del esquema proporcionado.
- Si la pregunta no se puede responder con los datos disponibles, devuelve 'NO SE PUEDE RESOLVER LA PETICIÓN' y nada más.

Guía para JOIN (cuando sea necesario):
- Utiliza las relaciones (claves foráneas) dadas en el esquema. Evita suposiciones.
- Prefiere JOIN explícitos (INNER/LEFT) con ON usando las PK↔FK documentadas.
- Evita CROSS JOIN salvo que se solicite explícitamente.
"""

# ------------------ Lectura de esquema enriquecido (tablas + vistas) ------------------

def read_schema(conn: sqlite3.Connection) -> dict:
    """
    Lee objetos (tablas y vistas), columnas (con PK en tablas) y FKs (solo tablas).
    Retorna:
    {
      "objname": {
        "kind": "table"|"view",
        "columns": [{"name": str, "type": str, "pk": bool}],
        "foreign_keys": [{"from": str, "table": str, "to": str}]  # vacío en vistas
      },
      ...
    }
    """
    schema = {}
    cur = conn.execute("""
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
    """)
    objects = cur.fetchall()
    for name, kind in objects:
        cols = conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        fks = conn.execute(f"PRAGMA foreign_key_list('{name}')").fetchall() if kind == "table" else []
        schema[name] = {
            "kind": kind,
            "columns": [{"name": c[1], "type": (c[2] or "").strip(), "pk": bool(c[5])} for c in cols],
            "foreign_keys": [{"from": fk[3], "table": fk[2], "to": fk[4]} for fk in fks],
        }
    return schema

def schema_to_text(schema: dict) -> str:
    """Esquema compacto para el LLM (tablas + vistas, PK y FK)."""
    if not schema:
        return "(BD sin tablas de usuario)"
    lines = []
    for obj in sorted(schema.keys(), key=lambda k: (schema[k]["kind"], k)):
        kind = schema[obj]["kind"]
        cols = schema[obj]["columns"]
        fks = schema[obj]["foreign_keys"] if kind == "table" else []
        fk_map = {fk["from"]: fk for fk in fks} if fks else {}

        col_texts = []
        for c in cols:
            name = c["name"]
            ctype = c["type"] or ""
            pk = " PK" if (kind == "table" and c["pk"]) else ""
            arrow = ""
            if kind == "table" and name in fk_map:
                target = fk_map[name]
                arrow = f" → {target['table']}.{target['to']}"
            col_texts.append(f"{name}{(' ' + ctype) if ctype else ''}{pk}{arrow}".strip())

        prefix = "[VIEW] " if kind == "view" else ""
        line = f"- {prefix}{obj}(" + ", ".join(col_texts) + ")"
        lines.append(line)

        if kind == "table" and fks:
            rels = ", ".join(f"{obj}.{fk['from']} → {fk['table']}.{fk['to']}" for fk in fks)
            lines.append(f"  joins: {rels}")
    return "\n".join(lines)

# ------------------ Few-shot: carga y selección de ejemplos ------------------

def load_examples(path: str|None) -> list[dict]:
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"No existe examples file: {p}")
    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        lines = raw.splitlines()
        i = (e.lineno or 1) - 1
        ctx = "\n".join(f"{n+1:>4}: {lines[n]}" for n in range(max(0,i-2), min(len(lines), i+3)))
        raise ValueError(f"JSON inválido en {p}: {e}\nContexto:\n{ctx}") from e
    out = []
    for ex in data:
        if "nl" in ex and "sql" in ex:
            out.append({
                "name": ex.get("name",""),
                "nl": ex["nl"].strip(),
                "sql": ex["sql"].strip()
            })
    return out

def _score_example(query: str, ex_nl: str) -> int:
    q = query.lower()
    e = ex_nl.lower()
    score = 0
    for kw in re.findall(r"\w+", q):
        if kw in e:
            score += 1
    return score

def pick_examples(query: str, examples: list[dict], k: int = 3) -> list[dict]:
    if not examples:
        return []
    scored = [(_score_example(query, ex["nl"]), ex) for ex in examples]
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = [ex for s, ex in scored if s > 0][:k]
    return chosen if chosen else examples[:k]

# ------------------ LLM y ejecución segura ------------------

def llm_generate_sql(client: OpenAI, schema_text: str, user_query: str, examples: list[dict] | None = None) -> str:
    """Pide al LLM la consulta SQL siguiendo las reglas, inyectando ejemplos si hay."""
    shots = []
    if examples:
        for ex in examples:
            shots.append({"role":"user","content": f"Pregunta (ejemplo): {ex['nl']}\nDevuelve SOLO la consulta SQL."})
            shots.append({"role":"assistant","content": ex["sql"]})

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
        {"role": "user", "content": f"Esquema de la base de datos (tablas/vistas/PK/FK):\n{schema_text}"},
        *shots,
        {"role": "user", "content": f"Pregunta del usuario (español):\n{user_query}\nDevuelve SOLO la consulta SQL (una línea)."}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    sql = resp.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = sql.rstrip("`").rstrip()
    sql = sql.splitlines()[0].strip()
    return sql

def validate_sql(sql: str):
    """Guardas de seguridad."""
    if SQL_MULTISTMT.search(sql):
        raise ValueError("Se detectó ';' o múltiples sentencias. Solo una sentencia SELECT está permitida.")
    if SQL_FORBIDDEN.search(sql):
        raise ValueError("Se detectó palabra clave no permitida (mutación/administración). Solo SELECT.")
    if not SQL_REQUIRE_SELECT.search(sql):
        raise ValueError("La consulta no es un SELECT.")
    if not SQL_REQUIRE_LIMIT.search(sql):
        sql += " LIMIT 100"
    return sql

def run_query(conn: sqlite3.Connection, sql: str):
    cur = conn.execute(sql)
    rows = cur.fetchall()
    headers = [d[0] for d in cur.description] if cur.description else []
    return headers, rows

# ------------------ Utilidades de inspección ------------------

def _connect_readonly(db_path: str) -> sqlite3.Connection:
    p = Path(db_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de BD: {p}")
    uri = f"file:{p.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def _print_schema_detail(conn: sqlite3.Connection, name: str):
    row = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE name=? AND type IN ('table','view')",
        (name,)
    ).fetchone()
    if not row:
        print(f"[ERR] No existe tabla o vista llamada '{name}'.")
        return
    kind = row["type"]

    cols = conn.execute(f"PRAGMA table_info('{name}')").fetchall()
    if not cols:
        print(f"[WARN] No se pudieron obtener columnas para '{name}'.")
        return

    headers = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
    print(f"\n[{kind.upper()}] {name}")
    print(tabulate(cols, headers=headers, tablefmt="github"))

    if kind == "table":
        fks = conn.execute(f"PRAGMA foreign_key_list('{name}')").fetchall()
        if fks:
            fk_headers = ["id", "seq", "table", "from", "to", "on_update", "on_delete", "match"]
            print("\nClaves foráneas:")
            print(tabulate(fks, headers=fk_headers, tablefmt="github"))
        else:
            print("\n(Sin claves foráneas)")

# ------------------ Helpers de gráficos ------------------

def _is_number(x):
    if x is None:
        return False
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def _is_datetime(x):
    if x is None:
        return False
    if isinstance(x, (datetime,)):
        return True
    try:
        dtparser.parse(str(x))
        return True
    except Exception:
        return False

def _to_datetime(x):
    if isinstance(x, datetime):
        return x
    return dtparser.parse(str(x))

def _choose_chart(headers, rows):
    if not headers or not rows:
        return None
    if len(headers) < 2:
        return None
    cols = list(zip(*rows))
    H = list(headers)
    max_cols = min(3, len(H))
    candidates = [(i, j) for i in range(max_cols) for j in range(max_cols) if i != j]
    for i, j in candidates:
        X = cols[i]; Y = cols[j]
        x_is_num = all(_is_number(v) for v in X if v is not None)
        y_is_num = all(_is_number(v) for v in Y if v is not None)
        x_is_dt  = all(_is_datetime(v) for v in X if v is not None)
        if x_is_dt and y_is_num:
            pairs = [(_to_datetime(a), float(b)) for a, b in zip(X, Y) if a is not None and _is_number(b)]
            if len(pairs) >= 2:
                pairs.sort(key=lambda t: t[0])
                xs, ys = zip(*pairs)
                return ("line", xs, ys, H[i], H[j])
        if x_is_num and y_is_num:
            xs = [float(v) for v in X if _is_number(v)]
            ys = [float(v) for v in Y if _is_number(v)]
            n = min(len(xs), len(ys))
            if n >= 2:
                return ("scatter", xs[:n], ys[:n], H[i], H[j])
        if (not x_is_num and not x_is_dt) and y_is_num:
            buckets = {}
            for a, b in zip(X, Y):
                if a is None or not _is_number(b):
                    continue
                buckets.setdefault(str(a), []).append(float(b))
            if len(buckets) >= 2:
                items = [(k, sum(v)/len(v)) for k, v in buckets.items()]
                items.sort(key=lambda t: t[1], reverse=True)
                items = items[:20]
                xs = [k for k, _ in items]
                ys = [v for _, v in items]
                return ("bar", xs, ys, H[i], f"avg({H[j]})")
    return None

def _ensure_interactive_backend():
    try:
        current = matplotlib.get_backend()
        if current.lower() == "agg":
            matplotlib.use("TkAgg")
    except Exception:
        pass

def _maybe_plot(headers, rows, force=False):
    if not rows or not headers:
        return False
    if len(rows) > 5000 and not force:
        return False
    _ensure_interactive_backend()
    decision = _choose_chart(headers, rows)
    if not decision and not force:
        return False
    try:
        plt.close("all")
        if decision:
            kind, xs, ys, xl, yl = decision
        else:
            xs = list(range(len(rows))); ys = list(range(len(rows)))
            xl, yl, kind = "x", "y", "line"
        if kind == "line":
            plt.plot(xs, ys)
        elif kind == "bar":
            plt.bar(range(len(xs)), ys)
            plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
        elif kind == "scatter":
            plt.scatter(xs, ys)
        else:
            return False
        plt.xlabel(xl); plt.ylabel(yl); plt.title("Vista rápida (auto)")
        plt.tight_layout(); plt.show()
        return True
    except Exception as e:
        print(f"[WARN] No se pudo mostrar gráfico: {e}")
        return False

# ------------------ Bucle interactivo ------------------

def interactive_loop(db_path: str, schema_file: str | None = None, examples_file: str | None = None):
    global CHARTS_MODE
    conn = _connect_readonly(db_path)

    # Log BD efectiva
    dblist = conn.execute("PRAGMA database_list").fetchall()
    print(f"[INFO] PRAGMA database_list -> {dblist}")

    # Esquema (dinámico vía PRAGMA en esta versión)
    schema = read_schema(conn)
    schema_text = schema_to_text(schema)

    # Cargar ejemplos (si se pasan)
    try:
        examples = load_examples(examples_file)
    except Exception as e:
        print(f"[WARN] No se pudieron cargar ejemplos: {e}")
        examples = []

    print("Agente NL→SQL para SQLite (solo lectura). Escribe 'salir' para terminar.")
    print(f"BD: {db_path}")
    print("Comandos: \\objects | \\tables | \\views | \\schema <nombre> | \\examples | \\chart on | \\chart off")
    print("Objetos disponibles:")
    print(schema_text)
    print("-" * 60)

    history = []

    while True:
        try:
            q = input("\nPregunta (español)> ").strip()
            if not q:
                continue
            low = q.lower()

            # Salir
            if low in {"salir", "exit", "quit"}:
                break

            # Charts toggle
            if low == "\\chart on":
                CHARTS_MODE = "auto"
                print("[INFO] Charts: auto")
                continue
            if low == "\\chart off":
                CHARTS_MODE = "off"
                print("[INFO] Charts: off")
                continue

            # Inspección esquema
            if low == "\\objects":
                print(schema_to_text(schema))
                continue
            if low == "\\tables":
                only_tables = {k: v for k, v in schema.items() if v["kind"] == "table"}
                print(schema_to_text(only_tables))
                continue
            if low == "\\views":
                only_views = {k: v for k, v in schema.items() if v["kind"] == "view"}
                print(schema_to_text(only_views))
                continue
            if low.startswith("\\schema "):
                name = q.split(None, 1)[1]
                _print_schema_detail(conn, name)
                continue

            # Ejemplos
            if low == "\\examples":
                if not examples:
                    print("[INFO] No hay ejemplos cargados. Usa --examples-file.")
                else:
                    for i, ex in enumerate(examples, 1):
                        print(f"{i}. {ex.get('name','(sin nombre)')}: {ex['nl']}")
                continue

            # --- NL preprocesado: quitar “gráfico” para no confundir al LLM ---
            wants_chart = bool(re.search(r"\bgr[áa]fico(s)?\b", low))
            nl_query = re.sub(r"\bgr[áa]fico(s)?\b", "", q, flags=re.IGNORECASE).strip()
            if (nl_query.startswith('"') and nl_query.endswith('"')) or (nl_query.startswith("'") and nl_query.endswith("'")):
                nl_query = nl_query[1:-1].strip()

            # Elegir ejemplos relevantes y generar SQL
            chosen = pick_examples(nl_query, examples, k=3)
            sql_raw = llm_generate_sql(client, schema_text, nl_query, examples=chosen)
            sql = validate_sql(sql_raw)

            # Ejecutar
            headers, rows = run_query(conn, sql)

            # Mostrar
            print("\nSQL:", sql)
            if rows:
                print(tabulate(rows, headers=headers, tablefmt="github"))
                print(f"\nFilas: {len(rows)}")
            else:
                print("(Sin resultados)")

            # Plot opcional
            if CHARTS_MODE == "auto":
                did = _maybe_plot(headers, rows, force=wants_chart)
                if did:
                    print("[INFO] Gráfico mostrado (ventana flotante).")

            history.append({"question": q, "sql": sql})
        except Exception as e:
            print(f"Error: {e}")

    # Cierre
    conn.close()
    with open("historial_nl_sql.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("\nHistorial guardado en historial_nl_sql.json")

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Agente NL→SQL para SQLite (solo SELECT).")
    ap.add_argument(
        "--db",
        default=r"C:\Users\jose.remartinez\bases_relacionales_ejemplos\SQLite\sakila_master.db",
        help="Ruta a la base de datos SQLite (por defecto: sakila_master.db)"
    )
    ap.add_argument(
        "--charts",
        choices=["auto","off"],
        default="auto",
        help="Mostrar gráficos automáticamente cuando sea posible (auto|off)."
    )
    ap.add_argument(
        "--examples-file",
        default=None,
        help="Ruta a un JSON con ejemplos few-shot (nl/sql)."
    )
    args = ap.parse_args()

    global CHARTS_MODE
    CHARTS_MODE = args.charts

    interactive_loop(args.db, examples_file=args.examples_file)

if __name__ == "__main__":
    main()
