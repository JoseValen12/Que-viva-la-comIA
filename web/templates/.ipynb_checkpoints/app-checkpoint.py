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
# RUTAS DE ARCHIVOS.
# -----------------------------
MODEL_PATH = "modelo.h5"
CLASS_NAMES_PATH = "class_names.json"
RECIPES_PATH = "procesado_total.ods"  # Indicamos el dataset de recetas en formato ODS.
DB_PATH = "data.db"

# -----------------------------
# FLASK APP.
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False  # Permitimos UTF-8 en respuestas JSON para no escapar acentos.

# -----------------------------
# CARGA MODELO + CLASES.
# -----------------------------
model = load_model(MODEL_PATH, compile=False)  # Cargamos el modelo sin recompilar para servir inferencias.
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)  # Cargamos la lista de clases en el mismo orden que el modelo.

# -----------------------------
# RECETAS (ODS) + SQLITE.
# -----------------------------
RECIPES_DF = None  # Diferimos la carga hasta la inicialización controlada.

def init_db():
    # Creamos tablas si no existen para registrar predicciones e ingredientes.
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
        # Añadimos catálogo de ingredientes seleccionados por el usuario.
        con.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            created_at INTEGER
        )
        """)
        con.commit()

def save_ingredient(name: str):
    # Insertamos el ingrediente si no existía previamente para mantener idempotencia.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO ingredients(name, created_at) VALUES(?, ?)",
            (name.strip(), int(time.time()))
        )
        con.commit()
        return cur.lastrowid

def delete_ingredient(name: str):
    # Eliminamos un ingrediente por nombre ignorando mayúsculas/minúsculas.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "DELETE FROM ingredients WHERE LOWER(name) = LOWER(?)",
            (name.strip(),)
        )
        con.commit()
        return cur.rowcount

def list_ingredients():
    # Listamos ingredientes guardados ordenados por fecha de creación descendente.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT id, name, created_at FROM ingredients ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [dict(id=r[0], name=r[1], created_at=r[2]) for r in rows]

def _normalize(text: str) -> str:
    # Normalizamos a ASCII, minúsculas y espaciado para comparaciones robustas.
    import re, unicodedata
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii","ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"[^a-z0-9% ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Catálogo fijo de alérgenos mapeado a sinónimos o ingredientes frecuentes.
# Ajustamos las claves a etiquetas presentes en el dataset para reducir falsos positivos.
ALLERGENS = {
    "gluten": [
        "Espaguetis", "Fideos", "Cuscus", "pan rallado", "pasta macarrones",
    ],
    "frutos_secos": [
        "Anacardos", "almendras", "avellanas", "Nueces", "pistacho", "piñones",
    ],
    "lactosa": [
        "Mantequilla", "Nata líquida", "leche desnatada", "leche entera",
        "leche semidesnatada", "queso fresco", "queso manchego", "yogur natural",
    ],
    "huevo": [
        "huevos", "Mayonesa", "unidad de huevo", "huevo,", "huevos," , "huevo", ",2 huevos,"
    ],
    "marisco": [
        "Gambas", "langostinos",
    ],
    "moluscos": [
        "Berberechos", "Mejillones", "almejas", "sepia", "calamares", "pulpo",
    ],
    "pescado": [
        "Salmon", "Merluza", "atun fresco", "bacalao", "boquerones", "sardinas",
    ],
    "apio": ["apio"],
    "mostaza": ["Mostaza"],
}

# Frases que implican ausencia explícita del alérgeno y que no deberían excluir la receta.
EXCEPTIONS = [
    "sin gluten", "gluten free", "libre de gluten",
    "sin lactosa", "lactose free", "libre de lactosa",
    "sin huevo", "egg free",
    "sin frutos secos", "nut free",
    "sin marisco", "sin moluscos", "sin pescado",
    "sin apio", "sin mostaza",
]

# Frases específicas para evitar coincidencias ambiguas que no son alérgeno real.
NEGATIVE_PHRASES = [
    "nuez moscada",
]

def exclude_allergies_df(df, selected_keys):
    """
    Excluimos recetas que contengan cualquier término asociado a las alergias seleccionadas
    buscando en columnas de texto clave, respetando excepciones de tipo 'sin X' y frases negativas.
    """
    import re
    if df is None or df.empty or not selected_keys:
        return df

    cols = [c for c in ["Ingredientes", "Lista de ingredientes", "Nombre", "Categoria"] if c in df.columns]
    if not cols:
        return df

    corpus = df[cols].astype(str).fillna("").agg(" ".join, axis=1)
    corpus_norm = corpus.map(_normalize)

    # Detectamos excepciones explícitas para no filtrar esas filas.
    if EXCEPTIONS:
        exceptions_rx = re.compile(r"|".join(map(re.escape, map(_normalize, EXCEPTIONS))))
        has_exception = corpus_norm.str.contains(exceptions_rx, na=False)
    else:
        has_exception = False

    # Detectamos frases negativas para minimizar falsos positivos.
    if NEGATIVE_PHRASES:
        neg_rx = re.compile(r"|".join(map(re.escape, map(_normalize, NEGATIVE_PHRASES))))
        has_negative = corpus_norm.str.contains(neg_rx, na=False)
    else:
        has_negative = False

    # Construimos patrón OR con todos los términos de las claves elegidas.
    terms = []
    for k in selected_keys:
        terms.extend(ALLERGENS.get(k, []))

    terms_norm = sorted({_normalize(t) for t in terms if t})
    if not terms_norm:
        return df

    # Coincidimos por palabra completa para reducir falsos positivos.
    allergen_rx = re.compile(r"\b(" + "|".join(map(re.escape, terms_norm)) + r")\b")

    has_allergen = corpus_norm.str.contains(allergen_rx, na=False)

    # Aplicamos la regla final combinando alérgenos, excepciones y frases negativas.
    if isinstance(has_exception, bool):
        safe_mask = ~(has_allergen) | (has_negative)
    else:
        safe_mask = ~(has_allergen & ~has_exception) | (has_negative)

    return df.loc[safe_mask]

def get_saved_ingredients():
    # Obtenemos todos los ingredientes guardados por el usuario.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT name FROM ingredients")
        return [r[0] for r in cur.fetchall()]

def recipes_for_ingredients_df(ingredients: list, limit: int = 20):
    # Buscamos recetas que contengan todos los ingredientes indicados en campos relevantes.
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
    # Limpiamos valores no finitos y NaN para serialización JSON estable.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    # Convertimos columnas numéricas a tipos nativos manejando no finitos.
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
    # Cargamos el ODS con engine odf, normalizamos nombres de columnas y validamos requeridas.
    df = pd.read_excel(RECIPES_PATH, engine="odf")
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Id", "Categoria", "Nombre"]:
        if col not in df.columns:
            raise RuntimeError(f"Falta la columna requerida en recetas: {col}")
    return df

def init_app_once():
    """
    Evitamos doble inicialización cuando Flask usa el reloader en modo debug.
    """
    global RECIPES_DF
    if getattr(app, "_inited", False):
        return
    init_db()
    RECIPES_DF = load_recipes()
    app._inited = True

# Inicializamos de forma segura considerando el reloader de Werkzeug.
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    init_app_once()

# -----------------------------
# PREPROCESO DE IMÁGENES.
# -----------------------------
def preprocess_image(file_storage):
    # Adaptamos el tamaño de la imagen a la entrada del modelo y aplicamos el preprocess de ResNetV2.
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
    # Extraemos las k clases más probables junto con sus puntuaciones.
    prob = np.array(prob).reshape(-1)
    idxs = np.argsort(prob)[::-1][:k]
    return [{"class": CLASS_NAMES[i], "score": float(prob[i])} for i in idxs]

# -----------------------------
# HELPERS PERSISTENCIA.
# -----------------------------
def save_prediction(image_name, top_class, score):
    # Persistimos una predicción para histórico y análisis posterior.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO predictions(image_name, top_class, score, created_at) VALUES(?,?,?,?)",
            (image_name or "", top_class, float(score or 0.0), int(time.time()))
        )
        con.commit()
        return cur.lastrowid

def delete_prediction(pred_id: int):
    # Eliminamos una predicción por id para limpieza del histórico.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        con.commit()
        return cur.rowcount

def list_predictions():
    # Listamos predicciones recientes ordenadas por fecha de creación.
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT id, image_name, top_class, score, created_at FROM predictions ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [dict(id=r[0], image_name=r[1], top_class=r[2], score=r[3], created_at=r[4]) for r in rows]

def saved_classes(distinct_only=True):
    # Obtenemos clases vistas previamente, con opción de deduplicar.
    with sqlite3.connect(DB_PATH) as con:
        if distinct_only:
            cur = con.execute("SELECT DISTINCT top_class FROM predictions")
            return [r[0] for r in cur.fetchall()]
        else:
            cur = con.execute("SELECT top_class FROM predictions")
            return [r[0] for r in cur.fetchall()]

def _to_builtin_scalar(v):
    # Convertimos escalares de NumPy a tipos nativos para JSON.
    if v is None or isinstance(v, bool): return v
    if isinstance(v, (np.integer,)):    return int(v)
    if isinstance(v, (np.floating,)):   return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_,)):      return bool(v)
    if isinstance(v, float):            return v if math.isfinite(v) else None
    return v

def json_sanitize(obj):
    # Normalizamos estructuras para ser serializables en JSON de forma segura.
    if obj is pd.NA: return None
    if isinstance(obj, pd.DataFrame):   return _df_records_clean(obj)
    if isinstance(obj, np.ndarray):     return [json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):           return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):  return [json_sanitize(x) for x in obj]
    if obj is None or isinstance(obj, bool): return obj
    if isinstance(obj, (np.integer,)):       return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj

# -----------------------------
# RECETAS + FILTROS.
# -----------------------------
def recipes_for_class(cls: str, limit: int = 20, ):
    # Recuperamos recetas cuya categoría coincida con la clase y, si falla, buscamos por nombre.
    df = RECIPES_DF
    if df is None:
        return []
    mask_exact = df["Categoria"].astype(str).str.strip().str.lower() == str(cls).strip().lower()
    out = df.loc[mask_exact]

    if out.empty:
        out = df[df["Nombre"].astype(str).str.contains(str(cls), case=False, na=False)]
    cols = [c for c in ["Id", "Categoria", "Nombre", "Valoracion", "Ingredientes", "Lista de ingredientes"] if c in df.columns]
    return _df_records_clean(out[cols].head(limit).reset_index(drop=True))

def recipes_from_saved(per_class: int = 5, distinct_classes: bool = True):
    # Para cada clase guardada, obtenemos un conjunto de recetas de referencia.
    classes = saved_classes(distinct_only=distinct_classes)
    result = {}
    for cls in classes:
        result[cls] = recipes_for_class(cls, limit=per_class)
    return result

# -----------------------------
# ENDPOINTS.
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    # Servimos la plantilla principal del frontend.
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Recibimos una imagen, preprocesamos, inferimos y guardamos la clase más probable como ingrediente.
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "Falta el campo 'image' (archivo)."}), 400

        x = preprocess_image(request.files["image"])
        pred = model.predict(x)
        probs = pred[0]

        # Normalizamos de forma defensiva por si el modelo no devuelve una distribución válida.
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0 or s > 1.5:
            e = np.exp(probs - np.max(probs))
            probs = e / np.sum(e)

        # Calculamos el top-k y seleccionamos la mejor predicción.
        top5 = top_k_predictions(probs, k=min(5, len(CLASS_NAMES)))
        selected = top5[0]["class"] if top5 else None

        return jsonify({"ok": True, "selected": selected, "top5": top5})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Guardamos una predicción simple sin asociar recetas.
@app.route("/predictions", methods=["POST"])
def create_prediction():
    """
    Cuerpo JSON esperado.
    {
      "image_name": "foto.jpg",
      "class": "NombreDeClase",
      "score": 0.97
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
    Cuerpo JSON.
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
    # Devolvemos el listado de ingredientes guardados.
    try:
        items = list_ingredients()
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Permitimos nombres con espacios o acentos usando <path:name> y codificación URL en el cliente.
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

@app.route("/predictions", methods=["GET"])
def get_predictions():
    # Exponemos las predicciones guardadas para depuración o UI.
    try:
        rows = list_predictions()
        return jsonify({"ok": True, "items": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Helper para construir diccionario de recetas desde ingredientes guardados.
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
    # Devolvemos la lista de claves de alérgenos con etiquetas legibles.
    data = [{"key": k, "label": k.replace("_", " ").title()} for k in ALLERGENS.keys()]
    return jsonify(data), 200

# Reemplazo de /my_recipes con filtro por alergias y unión de ingredientes.
@app.route("/my_recipes", methods=["GET", "POST"])
def my_recipes():
    try:
        per_class = int(request.args.get("per_class", "5"))

        # Leemos alergias desde POST JSON o desde parámetros de consulta en GET.
        selected_allergies = []
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            selected_allergies = data.get("allergies", []) or []
        else:
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
            # Obtenemos recetas candidatas para todos los ingredientes guardados.
            df = recipes_for_ingredients_df(ingrs, limit=per_class)

            # Aplicamos exclusión por alérgenos seleccionados.
            df = exclude_allergies_df(df, selected_allergies)

            # Serializamos registros limpios y usamos una clave descriptiva.
            recs = _df_records_clean(df)
            key = " + ".join(ingrs)
            data_out[key] = recs
            payload = {"ok": True, "data": data_out, "meta": meta}
        else:
            payload = {"ok": True, "data": {}, "meta": meta}

        # Sanitizamos y devolvemos respuesta JSON con codificación UTF-8.
        safe_payload = json_sanitize(payload)
        body = json.dumps(safe_payload, ensure_ascii=False, allow_nan=False)
        return Response(body, mimetype="application/json")

    except Exception as e:
        err = {"ok": False, "error": str(e)}
        return Response(json.dumps(err, ensure_ascii=False), mimetype="application/json", status=500)

# -----------------------------
# MAIN.
# -----------------------------
if __name__ == "__main__":
    # Iniciamos el servidor de desarrollo en localhost con recarga en debug.
    app.run(host="127.0.0.1", port=5000, debug=True)