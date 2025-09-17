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
- Incluye un limite razonable aunque el usuario no especifique límite.
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

# ------------------ Lectura de ejemplos (con embeddings) ------------------

def _load_examples(path: str | None) -> list[dict]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # formato esperado: [{"nl":"...","sql":"..."}, ...]
        out = []
        for e in data:
            if isinstance(e, dict) and "nl" in e and "sql" in e:
                out.append({"nl": str(e["nl"]), "sql": str(e["sql"])})
        return out
    except Exception as e:
        print(f"[WARN] No se pudieron leer ejemplos de {path}: {e}")
        return []

# ===== Embeddings OpenAI =====
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _ensure_example_embeddings(examples: list[dict], cache_path: str | None) -> list[dict]:
    """Añade clave 'emb' a cada ejemplo. Usa caché JSON si se proporciona cache_path."""
    if not examples:
        return []

    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    to_embed = []
    for ex in examples:
        nl = ex["nl"]
        emb = None
        if nl in cache:
            emb = cache[nl]
        if emb is None:
            to_embed.append(nl)
        else:
            ex["emb"] = emb

    if to_embed:
        new_embs = _embed_texts(to_embed)
        for nl, emb in zip(to_embed, new_embs):
            cache[nl] = emb
        # asignamos a los ejemplos que no tenían
        for ex in examples:
            if "emb" not in ex:
                ex["emb"] = cache.get(ex["nl"])  # puede ser None si algo falló
        # guardamos caché
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except Exception as e:
                print(f"[WARN] No se pudo guardar caché de embeddings: {e}")

    return examples


def _pick_examples_semantic(query: str, examples: list[dict], k: int = 3) -> list[dict]:
    if not examples:
        return []
    # Embedding de la query
    q_emb = _embed_texts([query])
    if not q_emb:
        return examples[:k]
    qv = q_emb[0]
    ranked = []
    for ex in examples:
        sim = 0.0
        if "emb" in ex and isinstance(ex["emb"], list):
            sim = _cosine_sim(qv, ex["emb"])
        ranked.append((sim, ex))
    ranked.sort(key=lambda t: t[0], reverse=True)
    top = [ex for s, ex in ranked if s > 0][:k]
    return top if top else [ex for _, ex in ranked][:k]

# ------------------ LLM y ejecución segura ------------------

def _examples_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    bloques = []
    for ex in examples:
        bloques.append(
            "Entrada:\n" + ex["nl"] + "\nSalida esperada:\n" + ex["sql"]
        )
    return "\n\nEjemplos:\n" + "\n\n".join(bloques)


def llm_generate_sql(client: OpenAI, schema_text: str, user_query: str, fewshots: list[dict] | None = None) -> str:
    shots_txt = _examples_block(fewshots or [])
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
        {
            "role": "user",
            "content": textwrap.dedent(f"""
            Esquema de la base de datos (tablas, vistas, columnas, PK y relaciones FK):
            {schema_text}

            {shots_txt}

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
        max_tokens=500,
    )
    sql = resp.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = sql.rstrip("`").rstrip()
    sql = sql.splitlines()[0].strip()
    return sql


def validate_sql(sql: str):
    """Aplica guardas de seguridad y añade LIMIT si falta."""
    if SQL_MULTISTMT.search(sql):
        raise ValueError("Se detectó ';' o múltiples sentencias. Solo una sentencia SELECT está permitida.")
    if SQL_FORBIDDEN.search(sql):
        raise ValueError("Se detectó palabra clave no permitida (mutación/administración). Solo SELECT.")
    if not SQL_REQUIRE_SELECT.search(sql):
        raise ValueError("La consulta no es un SELECT.")
    if not SQL_REQUIRE_LIMIT.search(sql):
        sql = sql + " LIMIT 500"
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

# ------------------ Helpers gráficos ------------------

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
        X = cols[i]
        Y = cols[j]
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
            xs = list(range(len(rows)))
            ys = list(range(len(rows)))
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

def interactive_loop(db_path: str, schema_file: str | None = None, examples_file: str | None = None):
    conn = _connect_readonly(db_path)

    dblist = conn.execute("PRAGMA database_list").fetchall()
    print(f"[INFO] PRAGMA database_list -> {dblist}")

    schema_dict = read_schema(conn)
    schema_text = schema_to_text(schema_dict)

    examples_all = _load_examples(examples_file)
    # Precalcular embeddings y caché (junto al fichero de ejemplos)
    cache_path = None
    if examples_file:
        cache_path = examples_file + ".embcache.json"
    examples_all = _ensure_example_embeddings(examples_all, cache_path)

    print("Agente NL→SQL para SQLite (solo lectura). Escribe 'salir' para terminar.")
    print(f"BD: {db_path}")
    print("Comandos: \\objects | \\tables | \\views | \\schema <nombre> | \\chart on | \\chart off")
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
                globals()["CHARTS_MODE"] = "auto"
                print("[INFO] Charts: auto")
                continue
            if low == "\\chart off":
                globals()["CHARTS_MODE"] = "off"
                print("[INFO] Charts: off")
                continue
            if low == "\\objects":
                print(schema_to_text(schema_dict))
                continue
            if low == "\\tables":
                only_tables = {k: v for k, v in schema_dict.items() if v["kind"] == "table"}
                print(schema_to_text(only_tables))
                continue
            if low == "\\views":
                only_views = {k: v for k, v in schema_dict.items() if v["kind"] == "view"}
                print(schema_to_text(only_views))
                continue
            if low.startswith("\\schema "):
                name = q.split(None, 1)[1]
                _print_schema_detail(conn, name)
                continue

            # Detección de intención de gráfico y limpieza de query
            wants_chart = bool(re.search(r"\bgr[áa]fico(s)?\b", low))
            nl_query = re.sub(r"\bgr[áa]fico(s)?\b", "", q, flags=re.IGNORECASE).strip()
            if (nl_query.startswith('"') and nl_query.endswith('"')) or (nl_query.startswith("'") and nl_query.endswith("'")):
                nl_query = nl_query[1:-1].strip()

            # Selección de few-shots con similitud semántica (top-K)
            fewshots = _pick_examples_semantic(nl_query, examples_all, k=3)

            # Generar SQL con few-shots
            sql_raw = llm_generate_sql(client, schema_text, nl_query, fewshots=fewshots)
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

            # Gráfico opcional
            if globals().get("CHARTS_MODE", "auto") == "auto":
                force_plot = wants_chart
                did = _maybe_plot(headers, rows, force=force_plot)
                if did:
                    print("[INFO] Gráfico mostrado (ventana flotante).")

            history.append({"question": q, "sql": sql, "used_examples": fewshots})
        except Exception as e:
            print(f"Error: {e}")

    conn.close()
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
    ap.add_argument(
        "--charts",
        choices=["auto","off"],
        default="auto",
        help="Mostrar gráficos automáticamente cuando sea posible (auto|off)."
    )
    ap.add_argument(
        "--examples-file",
        default=None,
        help="Ruta a un JSON con ejemplos few-shot (claves: nl, sql)."
    )
    ap.add_argument(
        "--schema-file",
        default=None,
        help="Ruta a un fichero .sql con el esquema (opcional, solo para referencia visual)."
    )

    args = ap.parse_args()

    globals()["CHARTS_MODE"] = args.charts

    # El esquema_file se muestra solo como referencia; el esquema efectivo se extrae de la BD
    if args.schema_file:
        try:
            with open(args.schema_file, "r", encoding="utf-8") as f:
                preview = f.read(4000)
            print("[INFO] Se cargó schema_file (vista previa 4k):\n" + preview[:4000])
        except Exception as e:
            print(f"[WARN] No se pudo leer {args.schema_file}: {e}")

    interactive_loop(args.db, args.schema_file, args.examples_file)


if __name__ == "__main__":
    main()
