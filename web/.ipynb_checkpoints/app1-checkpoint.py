from flask import Flask, request, jsonify, render_template, Response
import os, io, time, sqlite3, json
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
import re, unicodedata
import math

# -----------------------------
# RUTAS DE ARCHIVOS
# -----------------------------
MODEL_PATH = "modelo.h5"
CLASS_NAMES_PATH = "class_names.json"
RECIPES_PATH = "procesado_total.ods"  # dataset de recetas (.ods)
DB_PATH = "data.db"

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False

# -----------------------------
# CARGA MODELO + CLASES
# -----------------------------
model = load_model(MODEL_PATH, compile=False)
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

# -----------------------------
# RECETAS (ODS) + SQLITE
# -----------------------------
RECIPES_DF = None  # se llena en init_app()

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            top_class TEXT,
            score REAL,
            created_at INTEGER
        )
        """)
        # NUEVO: ingredientes
        con.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            created_at INTEGER
        )
        """)
        con.commit()
def save_ingredient(name: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO ingredients(name, created_at) VALUES(?, ?)",
            (name.strip(), int(time.time()))
        )
        con.commit()
        return cur.lastrowid

def delete_ingredient(name: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "DELETE FROM ingredients WHERE LOWER(name) = LOWER(?)",
            (name.strip(),)
        )
        con.commit()
        return cur.rowcount


def list_ingredients():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT id, name, created_at FROM ingredients ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [dict(id=r[0], name=r[1], created_at=r[2]) for r in rows]

def _normalize(text: str) -> str:
    import re, unicodedata
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii","ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"[^a-z0-9% ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Catálogo fijo: clave estable -> lista de términos/sinónimos
# --- DICCIONARIO DE ALÉRGENOS ADAPTADO A TUS ETIQUETAS ---
ALLERGENS = {
    # Se consideran con gluten por tus labels
    "gluten": [
        "Espaguetis", "Fideos", "Cuscus", "pan rallado", "pasta macarrones",
    ],

    # Árbol/frutos secos presentes en tus labels
    # (nota: "Nuez moscada" es EXCEPCIÓN abajo para evitar falsos positivos)
    "frutos_secos": [
        "Anacardos", "almendras", "avellanas", "Nueces", "pistacho", "piñones",
    ],

    # Lácteos presentes
    "lactosa": [
        "Mantequilla", "Nata líquida", "leche desnatada", "leche entera",
        "leche semidesnatada", "queso fresco", "queso manchego", "yogur natural",
    ],

    # Huevo y derivados (mayonesa suele contener huevo)
    "huevo": [
        "huevos", "Mayonesa", "unidad de huevo", "huevo"
    ],

    # Crustáceos (marisco) de tus etiquetas
    "marisco": [
        "Gambas", "langostinos",
    ],

    # Moluscos presentes
    "moluscos": [
        "Berberechos", "Mejillones", "almejas", "sepia", "calamares", "pulpo",
    ],

    # Pescado (no estaba en tu código original, pero es alérgeno UE)
    "pescado": [
        "Salmon", "Merluza", "atun fresco", "bacalao", "boquerones", "sardinas",
    ],

    # Otros alérgenos UE que sí aparecen en tus etiquetas
    "apio": ["apio"],
    "mostaza": ["Mostaza"],
}

# Frases que niegan la presencia del alérgeno (no se filtra si aparecen)
EXCEPTIONS = [
    "sin gluten", "gluten free", "libre de gluten",
    "sin lactosa", "lactose free", "libre de lactosa",
    "sin huevo", "egg free",
    "sin frutos secos", "nut free",
    "sin marisco", "sin moluscos", "sin pescado",
    "sin apio", "sin mostaza",
]

# Frases que evitarán falsos positivos por coincidencias parciales (p. ej. 'nuez moscada')
NEGATIVE_PHRASES = [
    "nuez moscada",
]

def exclude_allergies_df(df, selected_keys):
    """
    Excluye recetas que contengan CUALQUIER término de las alergias seleccionadas
    buscando en ["Ingredientes","Lista de ingredientes","Nombre","Categoria"].
    Respeta frases de excepción (p.ej., "sin gluten") y evita falsos positivos
    (p.ej., "nuez moscada").
    """
    import re
    if df is None or df.empty or not selected_keys:
        return df

    cols = [c for c in ["Ingredientes", "Lista de ingredientes", "Nombre", "Categoria"] if c in df.columns]
    if not cols:
        return df

    corpus = df[cols].astype(str).fillna("").agg(" ".join, axis=1)
    corpus_norm = corpus.map(_normalize)

    # 0) Si aparece una excepción, no filtramos esa fila
    if EXCEPTIONS:
        exceptions_rx = re.compile(r"|".join(map(re.escape, map(_normalize, EXCEPTIONS))))
        has_exception = corpus_norm.str.contains(exceptions_rx, na=False)
    else:
        has_exception = False

    # 1) Frases negativas (evitar falsos positivos)
    if NEGATIVE_PHRASES:
        neg_rx = re.compile(r"|".join(map(re.escape, map(_normalize, NEGATIVE_PHRASES))))
        has_negative = corpus_norm.str.contains(neg_rx, na=False)
    else:
        has_negative = False

    # 2) Construir patrón OR con los términos de las claves seleccionadas
    terms = []
    for k in selected_keys:
        terms.extend(ALLERGENS.get(k, []))

    # Normaliza y deduplica
    terms_norm = sorted({_normalize(t) for t in terms if t})
    if not terms_norm:
        return df

    # Palabras completas
    allergen_rx = re.compile(r"\b(" + "|".join(map(re.escape, terms_norm)) + r")\b")

    has_allergen = corpus_norm.str.contains(allergen_rx, na=False)

    # 3) Regla final:
    #    - Filtrar (excluir) si hay alérgeno
    #    - ...salvo si hay una EXCEPCIÓN explícita ("sin X")
    #    - ...y evitar falsos positivos donde la coincidencia aparezca dentro de NEGATIVE_PHRASES (p. ej. "nuez moscada")
    if isinstance(has_exception, bool):
        # Si no hay excepciones definidas
        safe_mask = ~(has_allergen) | (has_negative)  # no excluir si es negativo puro
    else:
        safe_mask = ~(has_allergen & ~has_exception) | (has_negative)

    return df.loc[safe_mask]


def get_saved_ingredients():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT name FROM ingredients")
        return [r[0] for r in cur.fetchall()]

def recipes_for_ingredients_df(ingredients: list, limit: int = 20):
    df = RECIPES_DF
    if df is None or not ingredients:
        return pd.DataFrame()

    parts = []
    for c in ["Ingredientes", "Lista de ingredientes", "Nombre", "Categoria"]:
        if c in df.columns:
            parts.append(df[c].astype(str))
    if not parts:
        return pd.DataFrame()

    corpus = parts[0]
    for s in parts[1:]:
        corpus = corpus.str.cat(s, sep=" ", na_rep="")

    norm_series = corpus.fillna("").map(_normalize)

    mask = pd.Series(True, index=df.index)
    for ing in ingredients:
        term = re.escape(_normalize(ing))
        mask &= norm_series.str.contains(term, na=False)

    out = df.loc[mask]
    cols = [c for c in ["Id", "Categoria", "Nombre", "Valoracion", "Ingredientes", "Lista de ingredientes", "Link_receta"] if c in df.columns]
    return out[cols].head(limit).reset_index(drop=True)




def _df_records_clean(df):
    # Reemplaza inf y NaN por None
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    # Para columnas numéricas, convierte NaN/no finitos a None con applymap
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].apply(lambda v: (None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v))

    def _to_builtin(v):
        if v is None or isinstance(v, (bool, np.bool_)): return bool(v) if v is not None else None
        if isinstance(v, (np.integer,)):                 return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            return f if math.isfinite(f) else None
        return v

    for c in df.columns:
        df[c] = df[c].map(_to_builtin)

    return df.to_dict(orient="records")




def load_recipes():
    # Requiere: pip install pandas odfpy
    df = pd.read_excel(RECIPES_PATH, engine="odf")
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Id", "Categoria", "Nombre"]:
        if col not in df.columns:
            raise RuntimeError(f"Falta la columna requerida en recetas: {col}")
    return df

def init_app_once():
    """
    Evita doble inicialización con el reloader de Flask en debug.
    """
    global RECIPES_DF
    if getattr(app, "_inited", False):
        return
    init_db()
    RECIPES_DF = load_recipes()
    app._inited = True

# Inicializa de forma segura
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    init_app_once()

# -----------------------------
# PREPROCESO DE IMÁGENES
# -----------------------------
def preprocess_image(file_storage):
    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    _, H, W, C = in_shape

    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((W, H))
    arr = np.asarray(img, dtype=np.float32)
    arr = resnet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def top_k_predictions(prob, k=5):
    prob = np.array(prob).reshape(-1)
    idxs = np.argsort(prob)[::-1][:k]
    return [{"class": CLASS_NAMES[i], "score": float(prob[i])} for i in idxs]

# -----------------------------
# HELPERS PERSISTENCIA
# -----------------------------
def save_prediction(image_name, top_class, score):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO predictions(image_name, top_class, score, created_at) VALUES(?,?,?,?)",
            (image_name or "", top_class, float(score or 0.0), int(time.time()))
        )
        con.commit()
        return cur.lastrowid

def delete_prediction(pred_id: int):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        con.commit()
        return cur.rowcount

def list_predictions():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT id, image_name, top_class, score, created_at FROM predictions ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [dict(id=r[0], image_name=r[1], top_class=r[2], score=r[3], created_at=r[4]) for r in rows]

def saved_classes(distinct_only=True):
    with sqlite3.connect(DB_PATH) as con:
        if distinct_only:
            cur = con.execute("SELECT DISTINCT top_class FROM predictions")
            return [r[0] for r in cur.fetchall()]
        else:
            cur = con.execute("SELECT top_class FROM predictions")
            return [r[0] for r in cur.fetchall()]

def _to_builtin_scalar(v):
    if v is None or isinstance(v, bool): return v
    if isinstance(v, (np.integer,)):    return int(v)
    if isinstance(v, (np.floating,)):   return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_,)):      return bool(v)
    if isinstance(v, float):            return v if math.isfinite(v) else None
    return v

def json_sanitize(obj):
    if obj is pd.NA: return None
    if isinstance(obj, pd.DataFrame):   return _df_records_clean(obj)
    if isinstance(obj, np.ndarray):     return [json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):           return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):  return [json_sanitize(x) for x in obj]
    # escalares
    if obj is None or isinstance(obj, bool): return obj
    if isinstance(obj, (np.integer,)):       return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj

# -----------------------------
# RECETAS + FILTROS
# -----------------------------
def recipes_for_class(cls: str, limit: int = 20, ):
    df = RECIPES_DF
    if df is None:
        return []
    # 1) Match exacto por Categoria
    mask_exact = df["Categoria"].astype(str).str.strip().str.lower() == str(cls).strip().lower()
    out = df.loc[mask_exact]

    # 2) Si vacío, buscar en Nombre
    if out.empty:
        out = df[df["Nombre"].astype(str).str.contains(str(cls), case=False, na=False)]
    # 4) Columnas a devolver
    cols = [c for c in ["Id", "Categoria", "Nombre", "Valoracion", "Ingredientes", "Lista de ingredientes"] if c in df.columns]
    return _df_records_clean(out[cols].head(limit).reset_index(drop=True))

def recipes_from_saved(per_class: int = 5, distinct_classes: bool = True):
    classes = saved_classes(distinct_only=distinct_classes)
    result = {}
    for cls in classes:
        result[cls] = recipes_for_class(cls, limit=per_class)
    return result


# -----------------------------
# ENDPOINTS
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "Falta el campo 'image' (archivo)."}), 400

        x = preprocess_image(request.files["image"])
        pred = model.predict(x)
        probs = pred[0]

        # Diagnóstico y normalización de seguridad
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0 or s > 1.5:
            e = np.exp(probs - np.max(probs))
            probs = e / np.sum(e)

        # Top-5 ya ordenado (asumiendo que top_k_predictions devuelve [{'class':..., 'score':...}, ...])
        top5 = top_k_predictions(probs, k=min(5, len(CLASS_NAMES)))

        # Seleccionar automáticamente la primera predicción
        selected = top5[0]["class"] if top5 else None

        # Guardar en BD de forma idempotente
        if selected:
            import sqlite3  # (o ponlo arriba del archivo)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO ingredients (name) VALUES (?)",
                    (selected,)
                )
                conn.commit()

        return jsonify({"ok": True, "selected": selected, "top5": top5})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# Guardar solo la predicción (no guarda recetas)
@app.route("/predictions", methods=["POST"])
def create_prediction():
    """
    Body JSON:
    {
      "image_name": "foto.jpg",  (opcional)
      "class": "NombreDeClase",  (requerido)
      "score": 0.97              (opcional)
    }
    """
    try:
        data = request.get_json(force=True)
        top_class = data.get("class")
        if not top_class:
            return jsonify({"ok": False, "error": "Falta 'class' en el body."}), 400

        image_name = data.get("image_name") or ""
        score = data.get("score")
        pred_id = save_prediction(image_name, top_class, score)
        return jsonify({"ok": True, "prediction_id": pred_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
@app.route("/ingredients", methods=["POST"])
def create_ingredient():
    """
    Body JSON:
    { "name": "tomate" }
    """
    try:
        data = request.get_json(force=True)
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Falta 'name'."}), 400
        new_id = save_ingredient(name)
        return jsonify({"ok": True, "ingredient": name, "created": bool(new_id)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/ingredients", methods=["GET"])
def get_ingredients():
    try:
        items = list_ingredients()
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Usamos <path:name> para soportar espacios/acentos (usa encodeURIComponent en el frontend)
@app.route("/ingredients/<path:name>", methods=["DELETE"])
def remove_ingredient(name):
    try:
        name = (name or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Falta nombre de ingrediente."}), 400
        deleted = delete_ingredient(name)
        if deleted == 0:
            return jsonify({"ok": False, "error": "No existe ese ingrediente."}), 404
        return jsonify({"ok": True, "deleted": name})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Listar lo guardado (para depurar/mostrar)
@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        rows = list_predictions()
        return jsonify({"ok": True, "items": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# --- helper nuevo ---
def recipes_from_ingredients(per_class: int = 5):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT name FROM ingredients")
        ingrs = [r[0] for r in cur.fetchall()]
    result = {}
    for ing in ingrs:
        result[ing] = recipes_for_class(ing, limit=per_class)
    return result

@app.get("/allergy_choices")
def allergy_choices():
    # Devuelve claves + etiqueta legible
    data = [{"key": k, "label": k.replace("_", " ").title()} for k in ALLERGENS.keys()]
    return jsonify(data), 200

# --- reemplazo de /my_recipes ---
@app.route("/my_recipes", methods=["GET", "POST"])
def my_recipes():
    try:
        per_class = int(request.args.get("per_class", "5"))

        # 1) Leer alergias seleccionadas
        #    - POST JSON: { "allergies": ["gluten","cacahuete"] }
        #    - GET: ?allergy=gluten&allergy=cacahuete  o  ?allergies=gluten,cacahuete
        selected_allergies = []
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            selected_allergies = data.get("allergies", []) or []
        else:
            # GET
            selected_allergies = request.args.getlist("allergy")
            if not selected_allergies:
                csv = request.args.get("allergies")
                if csv:
                    selected_allergies = [s.strip() for s in csv.split(",") if s.strip()]

        ingrs = get_saved_ingredients()

        data_out = {}
        meta = {
            "source": "ingredients_all",
            "per_class": per_class,
            "allergies": selected_allergies,
        }

        if ingrs:
            # 2) Obtener recetas como DF
            df = recipes_for_ingredients_df(ingrs, limit=per_class)

            # 3) Aplicar filtro por alergias
            df = exclude_allergies_df(df, selected_allergies)

            # 4) Serializar (limpio y consistente)
            recs = _df_records_clean(df)
            key = " + ".join(ingrs)
            data_out[key] = recs
            payload = {"ok": True, "data": data_out, "meta": meta}
        else:
            payload = {"ok": True, "data": {}, "meta": meta}


        safe_payload = json_sanitize(payload)
        body = json.dumps(safe_payload, ensure_ascii=False, allow_nan=False)
        return Response(body, mimetype="application/json")

    except Exception as e:
        err = {"ok": False, "error": str(e)}
        return Response(json.dumps(err, ensure_ascii=False), mimetype="application/json", status=500)
# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
