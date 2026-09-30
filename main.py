import os
import gdown
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. DESCARGAR CARPETA CON SUBCARPETAS DESDE DRIVE ---
DRIVE_FOLDER_ID = "1EiVkJHCb7M2K7AFmYbry8ty4Gk-acVLC"

os.makedirs("data", exist_ok=True)

if not os.path.exists("chroma_db_eval_recursive"):
    print("Descargando archivos y subcarpetas desde Google Drive...")
    try:
        # gdown descarga la carpeta y respeta la estructura de subcarpetas interna
        gdown.download_folder(id=DRIVE_FOLDER_ID, output="data", quiet=False, use_cookies=False)
        print("¡Todos los archivos y subcarpetas descargados con éxito!")
    except Exception as e:
        print(f"Error al descargar la carpeta: {e}")

# --- 2. INICIALIZAR EL MOTOR RAG ---
from checkpoint_5_1_hybrid_graph_flat import HybridEvaluationRetriever, HybridGraphAgent, make_llm

PDF_DIR = "./data"

app = FastAPI(title="Relant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Inicializando Motor RAG e indexando recursivamente...")
llm = make_llm()
retriever = HybridEvaluationRetriever(pdf_dir=PDF_DIR, top_k=10)
agent = HybridGraphAgent(llm, retriever)
print("Motor RAG listo.")

# --- 3. RUTAS DE LA API ---
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
    try:
        result = agent.run(req.question)
        return {
            "ok": True,
            "agent": result.get("answer", "No se generó respuesta."),
            "fixed": "(Pipeline fijo deshabilitado)",
            "steps": result.get("iterations", 0),
            "transcript": result.get("transcript", []),
            "used_external": any("web_paper_search" in t for t in result.get("transcript", [])),
            "image_urls": [],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/health")
async def health_check():
    return {"ok": True, "status": "online"}
