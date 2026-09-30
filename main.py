import os
import shutil
import zipfile
import gdown
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. DESCARGA Y EXTRAE EL ZIP DESDE GOOGLE DRIVE AL ARRANCAR
DRIVE_ZIP_ID = "1qOw8X3GZ-28KSTdhdroU9H8JxKjd6s-j"

if not os.path.exists("data"):
    print("Descargando archivo comprimido datos_rag.zip desde Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_ZIP_ID}"
    zip_path = "datos_rag.zip"
    
    try:
        gdown.download(url, zip_path, quiet=False)
        
        if os.path.exists(zip_path):
            print("Descomprimiendo estructura de archivos en el servidor...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            print("¡Archivos descomprimidos y listos!")
        else:
            print("Error: No se encontró el archivo ZIP tras la descarga.")
    except Exception as e:
        print(f"Error en la descarga o extracción desde Drive: {e}")

# Importamos tu motor RAG tras garantizar que la carpeta data exista
from checkpoint_5_1_hybrid_graph_flat import HybridEvaluationRetriever, HybridGraphAgent, make_llm, PDF_DIR

app = FastAPI(title="Relant RAG API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el motor RAG
print("Inicializando Motor RAG...")
llm = make_llm()
retriever = HybridEvaluationRetriever(pdf_dir=PDF_DIR, top_k=10)
agent = HybridGraphAgent(llm, retriever)
print("Motor RAG listo.")

# --- RUTAS DE LA INTERFAZ HTML Y API ---

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

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    allowed_extensions = {
        ".pdf", ".csv", ".xlsx", ".xls", ".pptx", ".ppt",
        ".docx", ".doc", ".png", ".jpg", ".jpeg", ".txt", ".md"
    }
    saved_files = []
    
    os.makedirs(PDF_DIR, exist_ok=True)
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            continue
            
        target_dir = Path(PDF_DIR)
        if ext in {".png", ".jpg", ".jpeg"}:
            target_dir = target_dir / "drawings"
            os.makedirs(target_dir, exist_ok=True)
            
        file_path = target_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        saved_files.append(file.filename)
        
    retriever._sync_and_index()
    return {"message": f"{len(saved_files)} archivos subidos e indexados con éxito.", "files": saved_files}

@app.get("/api/health")
async def health_check():
    return {"ok": True, "status": "online"}
