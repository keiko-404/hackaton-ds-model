import re
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.sparse import hstack

app = FastAPI(
    title="API Clasificador de Contenido Técnico",
    description="Clasifica contenido técnico y extrae palabras clave usando un modelo  de Regresión Logística.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #aqui dice que hay que poner el link de la empresa para darle permiso
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

modelo = joblib.load("modelo_clasificador.joblib")
vect_titulo = joblib.load("tfidf_titulo.joblib")
vect_texto = joblib.load("tfidf_texto.joblib")

PESO_TITULO = 2.0

NORMALIZACIONES = {
    r"\bc\+\+": "cplusplus",
    r"\bc#": "csharp",
    r"\bf#": "fsharp",
    r"\.net\b": "dotnet",
    r"\bnode\.js\b": "nodejs",
    r"\bvue\.js\b": "vuejs",
    r"\breact\.js\b": "reactjs",
    r"\bangular\.js\b": "angularjs",
    r"\bci\s*/\s*cd\b": "cicd",
}

def limpiar_texto_api(texto: str) -> str:
    texto = texto.lower()
    for patron, reemplazo in NORMALIZACIONES.items():
        texto = re.sub(patron, reemplazo, texto)
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúñü]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

class SolicitudClasificacion(BaseModel):
    titulo: str = Field(..., example="Curso de Python para análisis de datos")
    texto: str = Field(..., example="Aprende pandas, numpy, matplotlib y machine learning.")
    top_n_palabras: int = Field(5, ge=1, le=20, example=5)

class RespuestaClasificacion(BaseModel):
    categoria: str
    probabilidad: float
    informacion_adicional: list[str]

@app.get("/")
def inicio():
    return {"mensaje": "API de clasificación funcionando correctamente"}

@app.get("/health")
def health_check():
    esta_cargado = modelo is not None and vect_titulo is not None and vect_texto is not None

    return {
        "estado": "ok" if esta_cargado else "degradado",
        "modelo_cargado": esta_cargado
    }

@app.post("/predict", response_model=RespuestaClasificacion)
def clasificar(datos: SolicitudClasificacion):
    titulo_limpio = limpiar_texto_api(datos.titulo)
    descripcion_limpia = limpiar_texto_api(datos.texto)

    v_titulo = vect_titulo.transform([titulo_limpio]) * PESO_TITULO
    v_texto = vect_texto.transform([descripcion_limpia])
    vector = hstack([v_titulo, v_texto])

    categoria = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]
    probabilidad = round(float(probabilidades.max()), 4)

    nombres_features = (
        list(vect_titulo.get_feature_names_out())
        + list(vect_texto.get_feature_names_out())
    )

    vector_denso = vector.toarray()[0]
    indices_top = vector_denso.argsort()[::-1][:datos.top_n_palabras]
    palabras_clave = [
        nombres_features[i]
        for i in indices_top
        if vector_denso[i] > 0
    ]

    return {
        "categoria": categoria,
        "probabilidad": probabilidad,
        "informacion_adicional": palabras_clave,
    }