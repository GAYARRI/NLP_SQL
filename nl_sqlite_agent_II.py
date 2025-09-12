#!/usr/bin/env python3
# nl_sqlite_agent.py
# Agente NL→SQL seguro para SQLite (solo lectura) con PK/FK, relaciones, vistas y \schema

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import math
from datetime import datetime
from dateutil import parser as dtparser  # pip install python-dateutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib; print(matplotlib.get_backend())



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

# --- Añadido Soporte CLI y comandos ---------------------------------------    




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


# ------------------ Helpers Punto2-------------


# ---------- Detección de tipos ----------
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
    # Intento de parseo “suave”
    try:
        dtparser.parse(str(x))
        return True
    except Exception:
        return False

def _to_datetime(x):
    if isinstance(x, datetime):
        return x
    return dtparser.parse(str(x))

# ---------- Elección de gráfico ----------
def _choose_chart(headers, rows):
    """
    Decide un tipo de gráfico y qué columnas usar.
    Retorna (kind, x_vals, y_vals, x_label, y_label) o None si no procede.
    Reglas:
      - 2 columnas numéricas -> scatter
      - 1 datetime + 1 numérico -> line (ordenado por fecha)
      - 1 categórica + 1 numérico -> bar (top N categorías)
    """
    if not headers or not rows:
        return None
    if len(headers) < 2:
        return None

    # Tomamos como base las dos primeras columnas útiles
    cols = list(zip(*rows))  # columnas
    H = list(headers)

    # Buscamos pares viables (hasta 3 primeras columnas)
    max_cols = min(3, len(H))
    candidates = [(i, j) for i in range(max_cols) for j in range(max_cols) if i != j]

    for i, j in candidates:
        X = cols[i]
        Y = cols[j]
        x_is_num = all(_is_number(v) for v in X if v is not None)
        y_is_num = all(_is_number(v) for v in Y if v is not None)
        x_is_dt  = all(_is_datetime(v) for v in X if v is not None)

        # (A) datetime + numérico -> línea temporal
        if x_is_dt and y_is_num:
            pairs = [(_to_datetime(a), float(b)) for a, b in zip(X, Y) if a is not None and _is_number(b)]
            if len(pairs) >= 2:
                pairs.sort(key=lambda t: t[0])
                xs, ys = zip(*pairs)
                return ("line", xs, ys, H[i], H[j])

        # (B) dos numéricos -> dispersión
        if x_is_num and y_is_num:
            xs = [float(v) for v in X if _is_number(v)]
            ys = [float(v) for v in Y if _is_number(v)]
            n = min(len(xs), len(ys))
            if n >= 2:
                return ("scatter", xs[:n], ys[:n], H[i], H[j])

        # (C) categórica + numérico -> barras (agregamos media por categoría)
        if (not x_is_num and not x_is_dt) and y_is_num:
            buckets = {}
            for a, b in zip(X, Y):
                if a is None or not _is_number(b):
                    continue
                buckets.setdefault(str(a), []).append(float(b))
            if len(buckets) >= 2:
                items = [(k, sum(v)/len(v)) for k, v in buckets.items()]
                # top 20 categorías
                items.sort(key=lambda t: t[1], reverse=True)
                items = items[:20]
                xs = [k for k, _ in items]
                ys = [v for _, v in items]
                return ("bar", xs, ys, H[i], f"avg({H[j]})")

    return None

def _ensure_interactive_backend():
    """Intenta asegurar un backend interactivo; si no, informa."""
    try:
        current = matplotlib.get_backend()
        # Si estamos en modo 'Agg' (no interactivo), intentamos TkAgg
        if current.lower() == "agg":
            matplotlib.use("TkAgg")
    except Exception:
        pass  # si falla, seguimos; plt.show() lo intentará igualmente

def _maybe_plot(headers, rows, force=False):
    """
    Intenta dibujar un gráfico en ventana flotante si hay datos adecuados
    o si se fuerza con 'force=True'.
    """
    if not rows or not headers:
        return False
    # límite prudente para no bloquear con datos enormes
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
            # Modo “fuerza bruta”: si hay 2 columnas, intenta scatter/line simple
            xs = list(range(len(rows)))
            ys = list(range(len(rows)))
            xl, yl, kind = "x", "y", "line"

        if kind == "line":
            plt.plot(xs, ys)
        elif kind == "bar":
            # Para barras: rotamos etiquetas si son largas
            plt.bar(range(len(xs)), ys)
            plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
        elif kind == "scatter":
            plt.scatter(xs, ys)
        else:
            return False

        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title("Vista rápida (auto)")
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"[WARN] No se pudo mostrar gráfico: {e}")
        return False











# ------------------ Bucle interactivo ------------------

def interactive_loop(db_path: str):
    # Conexión SQLite (solo lectura)
    global CHARTS_MODE
    conn = _connect_readonly(db_path)

    # Log BD efectiva
    dblist = conn.execute("PRAGMA database_list").fetchall()
    print(f"[INFO] PRAGMA database_list -> {dblist}")

    # Lee y renderiza esquema enriquecido
    schema = read_schema(conn)
    schema_text = schema_to_text(schema)

    print("Agente NL→SQL para SQLite (solo lectura). Escribe 'salir' para terminar.")
    print(f"BD: {db_path}")
    print("Comandos: \\objects | \\tables | \\views | \\schema <nombre> | \\chart on | \\chart off")   #### Añadido 4   
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
            if low == "\\chart on":
                CHARTS_MODE = "auto"
                print("[INFO] Charts: auto")
                continue
            if low == "\\chart off":
                CHARTS_MODE = "off"
                print("[INFO] Charts: off")
                continue
            # --- detectar intención de gráfico y limpiar query para el LLM ---
            wants_chart = bool(re.search(r"\bgr[áa]fico(s)?\b", low))
            nl_query = re.sub(r"\bgr[áa]fico(s)?\b", "", q, flags=re.IGNORECASE).strip()
            #    si el usuario puso comillas alrededor de toda la frase, las quitamos
            if (nl_query.startswith('"') and nl_query.endswith('"')) or (nl_query.startswith("'") and nl_query.endswith("'")):
             nl_query = nl_query[1:-1].strip()

            # Generar SQL (usar la pregunta LIMPIA)
            sql_raw = llm_generate_sql(client, schema_text, nl_query)
            sql = validate_sql(sql_raw)


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

##############   Añadido 3 3ºPunto##############################################################

            low = q.lower()

            if low in {"salir", "exit", "quit"}:
                break
            if low == "\\chart on":
                CHARTS_MODE = "auto"
                print("[INFO] Charts: auto")
                continue
            if low == "\\chart off":
                CHARTS_MODE = "off"
                print("[INFO] Charts: off")
                continue
     
###################################################################################################

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


            ########################## plot opcional ###################################################

            if CHARTS_MODE == "auto":
               # Si el usuario escribió la palabra "grafico" en su pregunta, forzamos
               force_plot = ("grafico" in low or "gráfico" in low)
               did = _maybe_plot(headers, rows, force=force_plot)
               if did:
                  print("[INFO] Gráfico mostrado (ventana flotante).")

            ##############################################################################################
            
            history.append({"question": q, "sql": sql})
        except Exception as e:
            print(f"Error: {e}")

    # Cierre
    conn.close()
    # Guarda historial básico
    with open("historial_nl_sql.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("\nHistorial guardado en historial_nl_sql.json")

# ------------------ Añadido 3 al main ------------------

def main():
    
    ap = argparse.ArgumentParser(description="Agente NL→SQL para SQLite (solo SELECT).")
    ap.add_argument(
        "--db",
        default="C:\\Users\\jose.remartinez\\bases_relacionales_ejemplos\\SQLite\\sakila_master.db",
        help="Ruta a la base de datos SQLite (por defecto: sakila_master.db)"
    )
    ap.add_argument(
        "--charts",
        choices=["auto","off"],
        default="auto",
        help="Mostrar gráficos automáticamente cuando sea posible (auto|off)."
    )
    args = ap.parse_args()

    global CHARTS_MODE
    CHARTS_MODE = args.charts

    interactive_loop(args.db)


if __name__ == "__main__":
    main()
