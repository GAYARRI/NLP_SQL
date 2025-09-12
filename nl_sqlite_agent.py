#!/usr/bin/env python3
# nl_sqlite_agent.py
# Agente NL→SQL seguro para SQLite (solo lectura) con PK/FK, relaciones, vistas y \schema

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os

# Cargar .env: primero desde el cwd y, si no aparece, el .env junto al script
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = Path(__file__).resolve().with_name(".env")
# Importante: permitir que .env sobrescriba valores previos (placeholders, etc.)
load_dotenv(dotenv_path=dotenv_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(f"No se encontró OPENAI_API_KEY. Revisa tu .env en: {dotenv_path}")

# Cliente OpenAI (se crea una vez, aquí)
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
- No uses subconsultas peligrosas que escriban datos (no permitidas en SQLite).
- NUNCA incluyas ';' al final.
- Incluye siempre LIMIT razonable (por defecto 100) si el usuario no especifica límite.
- Usa exclusivamente tablas/columnas del esquema proporcionado.
- Emplea parámetros literales directos (sin variables).
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
        # En vistas, PRAGMA foreign_key_list devuelve vacío (vistas no tienen FKs propias)
        fks = conn.execute(f"PRAGMA foreign_key_list('{name}')").fetchall() if kind == "table" else []
        schema[name] = {
            "kind": kind,
            "columns": [{"name": c[1], "type": (c[2] or "").strip(), "pk": bool(c[5])} for c in cols],
            "foreign_keys": [{"from": fk[3], "table": fk[2], "to": fk[4]} for fk in fks],
        }
    return schema

def schema_to_text(schema: dict) -> str:
    """
    Representa el esquema en texto compacto y legible por el LLM, incluyendo PK y FK.
    Para vistas, se antepone [VIEW] y no se listan 'joins:' (no aplican PK/FK).
    """
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
            if ctype:
                col_texts.append(f"{name} {ctype}{pk}{arrow}".strip())
            else:
                col_texts.append(f"{name}{pk}{arrow}".strip())

        prefix = "[VIEW] " if kind == "view" else ""
        line = f"- {prefix}{obj}(" + ", ".join(col_texts) + ")"
        lines.append(line)

        if kind == "table" and fks:
            rels = ", ".join(f"{obj}.{fk['from']} → {fk['table']}.{fk['to']}" for fk in fks)
            lines.append(f"  joins: {rels}")

    return "\n".join(lines)

# ------------------ LLM y ejecución segura ------------------

def llm_generate_sql(client: OpenAI, schema_text: str, user_query: str) -> str:
    """Pide al LLM la consulta SQL siguiendo las reglas."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
        {
            "role": "user",
            "content": textwrap.dedent(f"""
            Esquema de la base de datos (tablas, vistas, columnas, PK y relaciones FK):
            {schema_text}

            Pregunta del usuario (español):
            {user_query}

            Devuelve SOLO la consulta SQL (una línea).
            """).strip(),
        },
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    sql = resp.choices[0].message.content.strip()
    # En caso de que el modelo devuelva código en bloque, lo limpiamos.
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = sql.rstrip("`").rstrip()
    # Primera línea únicamente
    sql = sql.splitlines()[0].strip()
    return sql

def validate_sql(sql: str):
    """Aplica guardas de seguridad."""
    if SQL_MULTISTMT.search(sql):
        raise ValueError("Se detectó ';' o múltiples sentencias. Solo una sentencia SELECT está permitida.")
    if SQL_FORBIDDEN.search(sql):
        raise ValueError("Se detectó palabra clave no permitida (mutación/administración). Solo SELECT.")
    if not SQL_REQUIRE_SELECT.search(sql):
        raise ValueError("La consulta no es un SELECT.")
    if not SQL_REQUIRE_LIMIT.search(sql):
        # Añade LIMIT 100 si faltase
        sql += " LIMIT 100"
    return sql

def run_query(conn: sqlite3.Connection, sql: str):
    cur = conn.execute(sql)
    rows = cur.fetchall()
    headers = [d[0] for d in cur.description] if cur.description else []
    return headers, rows

# ------------------ Utilidades de inspección ------------------

def _connect_readonly(db_path: str) -> sqlite3.Connection:
    # Normaliza y valida la ruta
    p = Path(db_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de BD: {p}")
    # Conexión solo lectura vía URI (evita crear BD vacías)
    uri = f"file:{p.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def _print_schema_detail(conn: sqlite3.Connection, name: str):
    # ¿Existe y qué tipo es?
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

# ------------------ Bucle interactivo ------------------

def interactive_loop(db_path: str):
    # Conexión SQLite (solo lectura)
    conn = _connect_readonly(db_path)

    # Log BD efectiva
    dblist = conn.execute("PRAGMA database_list").fetchall()
    print(f"[INFO] PRAGMA database_list -> {dblist}")

    # Lee y renderiza esquema enriquecido
    schema = read_schema(conn)
    schema_text = schema_to_text(schema)

    print("Agente NL→SQL para SQLite (solo lectura). Escribe 'salir' para terminar.")
    print(f"BD: {db_path}")
    print("Comandos: \\objects | \\tables | \\views | \\schema <nombre>")
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

            # Comandos de utilidad
            if low in {"salir", "exit", "quit"}:
                break
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

            # Generar SQL
            sql_raw = llm_generate_sql(client, schema_text, q)
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

            history.append({"question": q, "sql": sql})
        except Exception as e:
            print(f"Error: {e}")

    # Cierre
    conn.close()
    # Guarda historial básico
    with open("historial_nl_sql.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("\nHistorial guardado en historial_nl_sql.json")

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Agente NL→SQL para SQLite (solo SELECT).")
    ap.add_argument(
        "--db",
        default="C:\\Users\\jose.remartinez\\bases_relacionales_ejemplos\\SQLite\\sakila_master.db",
        help="Ruta a la base de datos SQLite (por defecto: sakila_master.db)"
    )
    args = ap.parse_args()
    interactive_loop(args.db)

if __name__ == "__main__":
    main()
