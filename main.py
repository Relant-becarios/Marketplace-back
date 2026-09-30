import os
import shutil
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importamos tu motor RAG (Asegúrate de haberle quitado el " (1)" al nombre del archivo)
from backend.checkpoint_5_1_hybrid_graph_flat import HybridEvaluationRetriever, HybridGraphAgent, make_llm, PDF_DIR

app = FastAPI(title="Relant RAG API")

# 1. Configuración de CORS para que tu Vue pueda conectarse sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Inicializar el motor RAG al arrancar el servidor
print("Inicializando Motor RAG...")
llm = make_llm()
retriever = HybridEvaluationRetriever(pdf_dir=PDF_DIR, top_k=10)
agent = HybridGraphAgent(llm, retriever)
print("Motor RAG listo.")

# --- RUTAS DE LA INTERFAZ HTML ---

@app.get("/")
@app.get("/asistente")
async def serve_interfaz():
    """Servir directamente el archivo HTML al navegador."""
    html_path = Path("interfaz.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Archivo interfaz.html no encontrado. Asegúrate de que esté en la misma carpeta que main.py.")
    return FileResponse(html_path)

# --- RUTAS DE LA API (RAG Y ARCHIVOS) ---

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    """Procesa la pregunta en el grafo LangGraph y devuelve la respuesta."""
    try:
        result = agent.run(req.question)
        return {
            "ok": True,
            "agent": result.get("answer", "No se generó respuesta."),
            "fixed": "(La vista de pipeline fijo está deshabilitada, viendo solo el Agente)",
            "steps": result.get("iterations", 0),
            "transcript": result.get("transcript", []),
            "used_external": any("web_paper_search" in t or "calculator" in t for t in result.get("transcript", [])),
            "image_urls": [], # Aquí se poblarían las imágenes si el agente retorna URLs
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Recibir múltiples archivos desde un frontend."""
    allowed_extensions = {".pdf", ".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".txt", ".md"}
    saved_files = []
    
    os.makedirs(PDF_DIR, exist_ok=True)
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            continue
            
        target_dir = Path(PDF_DIR)
        # Los dibujos deben ir a la subcarpeta 'drawings' según tu script
        if ext in {".png", ".jpg", ".jpeg"}:
            target_dir = target_dir / "drawings"
            os.makedirs(target_dir, exist_ok=True)
            
        file_path = target_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        saved_files.append(file.filename)
        
    # Reindexar la colección después de subir archivos
    retriever._sync_and_index()
    return {"message": f"{len(saved_files)} archivos subidos e indexados.", "files": saved_files}

@app.get("/api/health")
async def health_check():
    """Para que la interfaz web sepa que el servidor está conectado."""
    return {"ok": True, "status": "online"}