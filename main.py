import os
import zipfile
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Variables globales para el RAG
retriever = None
agent = None

# ID de tu archivo ZIP directo de Google Drive
DRIVE_ZIP_ID = "1qOw8X3GZ-28KSTdhdroU9H8JxKjd6s-j"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, agent
    print("🚀 Servidor iniciado. Verificando datos y RAG...")
    
    # 1. Descarga e inicio en segundo plano tras abrir el puerto
    if not os.path.exists("data"):
        print("Descargando archivo comprimido datos_rag.zip desde Google Drive...")
        zip_path = "datos_rag.zip"
        download_url = f"https://docs.google.com/uc?export=download&confirm=t&id={DRIVE_ZIP_ID}"
        
        try:
            cmd = ["curl", "-L", "-o", zip_path, download_url]
            subprocess.run(cmd, check=True)
            
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000:
                print("Descomprimiendo archivos en el servidor...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove(zip_path)
                print("¡Archivos descomprimidos y listos!")
            else:
                print("Error: El archivo descargado está vacío o es inválido.")
        except Exception as e:
            print(f"Error en la descarga o extracción desde Drive: {e}")

    # 2. Inicialización del motor RAG
    try:
        from checkpoint_5_1_hybrid_graph_flat import HybridEvaluationRetriever, HybridGraphAgent, make_llm, PDF_DIR
        print("Inicializando Motor RAG...")
        llm = make_llm()
        retriever = HybridEvaluationRetriever(pdf_dir=PDF_DIR, top_k=10)
        agent = HybridGraphAgent(llm, retriever)
        print("✅ Motor RAG cargado y listo para consultas.")
    except Exception as e:
        print(f"Error al inicializar el RAG: {e}")

    yield
    print("Servidor apágandose...")

app = FastAPI(title="Relant RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RUTAS DE LA API ---

@app.get("/")
@app.get("/asistente")
async def serve_interfaz():
    html_path = Path("interfaz.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Archivo interfaz.html no encontrado.")
    return FileResponse(html_path)

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    if agent is None:
        return {"ok": False, "error": "El motor RAG aún se está inicializando en el servidor. Por favor reintenta en un momento."}
    try:
        result = agent.run(req.question)
        return {
            "ok": True,
            "agent": result.get("answer", "No se generó respuesta."),
            "fixed": "(Pipeline fijo deshabilitado en producción)",
            "steps": result.get("iterations", 0),
            "transcript": result.get("transcript", []),
            "used_external": any("web_paper_search" in t or "calculator" in t for t in result.get("transcript", [])),
            "image_urls": [],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/health")
async def health_check():
    return {"ok": True, "status": "online" if agent is not None else "initializing"}
