import os
import zipfile
import gdown
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. DESCARGA RÁPIDA DEL ZIP EN LUGAR DE CARPETA ---
# Sustituye este ID por el ID exacto de tu archivo datos_rag.zip en Drive
DRIVE_ZIP_ID = "11lJDvthCF2dZXE2_q8kiJOLk8dNdADU6"

if not os.path.exists("data"):
    print("Descargando archivo comprimido datos_rag.zip desde Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_ZIP_ID}"
    zip_path = "datos_rag.zip"
    
    try:
        # fuzzy=True salta la confirmación de escaneo de virus de Google para archivos grandes
        gdown.download(url, zip_path, quiet=False, fuzzy=True)
        
        if os.path.exists(zip_path):
            print("Descomprimiendo archivos en el servidor...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            print("¡Archivos descomprimidos y listos!")
        else:
            print("Error: No se encontró el archivo ZIP tras la descarga.")
    except Exception as e:
        print(f"Error en la descarga/extracción: {e}")

# --- 2. INICIALIZAR MOTOR RAG ---
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

print("Inicializando Motor RAG...")
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
    return {"ok": True, "status": "online"}
