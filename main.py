from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Document, DocumentChunk, Ticket
from llm import generate_knowledge_answer, generate_search_keyword
from typing import Literal
from pathlib import Path
from uuid import uuid4

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DocFlow Agent",
    description="企业知识库与工单协同 Agent",
    version="0.1.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".md"}

def split_text(text: str, chunk_size: int = 200) -> list[str]:
    text = text.strip()

    return [
        text[index:index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ]

class TicketCreate(BaseModel):
    title: str
    content: str
    priority: str = "normal"

class TicketStatusUpdate(BaseModel):
    status: Literal["open", "processing", "closed"]

class KnowledgeAsk(BaseModel):
    question: str

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "DocFlow Agent API is running",
    }


@app.post("/tickets")
def create_ticket(ticket: TicketCreate, db: Session = Depends(get_db)):
    new_ticket = Ticket(**ticket.model_dump())

    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)

    return {
        "message": "工单创建成功",
        "ticket": new_ticket,
    }


@app.get("/tickets")
def list_tickets(db: Session = Depends(get_db)):
    ticket_items = db.query(Ticket).order_by(Ticket.id.desc()).all()

    return {
        "total": len(ticket_items),
        "items": ticket_items,
    }

@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="工单不存在",
        )

    return ticket

@app.patch("/tickets/{ticket_id}/status")
def update_ticket_status(
    ticket_id: int,
    ticket_data: TicketStatusUpdate,
    db: Session = Depends(get_db),
):
    ticket = db.get(Ticket, ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="工单不存在",
        )

    ticket.status = ticket_data.status

    db.commit()
    db.refresh(ticket)

    return {
        "message": "工单状态更新成功",
        "ticket": ticket,
    }
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="请选择要上传的文件",
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="当前仅支持 .txt 和 .md 文件",
        )

    saved_filename = f"{uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / saved_filename

    content = await file.read()
    file_path.write_bytes(content)

    try:
       text_content = content.decode("utf-8")
    except UnicodeDecodeError:
       raise HTTPException(
        status_code=400,
        detail="文件编码必须是 UTF-8",
    )

    document = Document(
        original_filename=file.filename,
        saved_filename=saved_filename,
        file_size=len(content),
    )

    db.add(document)
    db.commit()
    db.refresh(document)
    chunks = split_text(text_content)

    for index, chunk_content in enumerate(chunks):
        chunk = DocumentChunk(
           document_id=document.id,
           chunk_index=index,
           content=chunk_content,
    )
        db.add(chunk)

    db.commit()
    return {
        "message": "文档上传成功",
        "document": document,
    }
@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.id.desc()).all()

    return {
        "total": len(documents),
        "items": documents,
    }

@app.get("/documents/{document_id}/chunks")
def list_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="文档不存在",
        )

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    return {
        "document_id": document_id,
        "total": len(chunks),
        "items": chunks,
    }

@app.get("/documents/search")
def search_document_chunks(
    query: str,
    db: Session = Depends(get_db),
):
    keyword = query.strip()

    if not keyword:
        raise HTTPException(
            status_code=400,
            detail="查询关键词不能为空",
        )

    results = (
        db.query(DocumentChunk, Document.original_filename)
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .filter(DocumentChunk.content.contains(keyword))
        .order_by(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
        )
        .all()
    )

    items = [
        {
            "document_id": chunk.document_id,
            "document_name": original_filename,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk, original_filename in results
    ]

    return {
        "query": keyword,
        "total": len(items),
        "items": items,
    }

@app.post("/knowledge/ask")
def ask_knowledge_base(
    request: KnowledgeAsk,
    db: Session = Depends(get_db),
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空",
    )

    keyword = generate_search_keyword(question)


    results = (
        db.query(DocumentChunk, Document.original_filename)
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .filter(DocumentChunk.content.contains(keyword))
        .order_by(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
        )
        .limit(3)
        .all()
    )

    if not results:
        fallback_keyword = keyword[:2]

    if len(fallback_keyword) >= 2:
        results = (
            db.query(DocumentChunk, Document.original_filename)
            .join(
                Document,
                DocumentChunk.document_id == Document.id,
            )
            .filter(DocumentChunk.content.contains(fallback_keyword))
            .order_by(
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
            )
            .limit(3)
            .all()
        )

        if results:
            keyword = fallback_keyword

    if not results:
        return {
        "keyword": keyword,
        "answer": "知识库中没有找到相关资料。",
        "sources": [],
    }

    context_parts = []
    sources = []

    for chunk, original_filename in results:
        context_parts.append(
            f"文档：{original_filename}\n"
            f"内容：{chunk.content}"
        )

        sources.append(
            {
                "document_id": chunk.document_id,
                "document_name": original_filename,
                "chunk_index": chunk.chunk_index,
            }
        )

    answer = generate_knowledge_answer(
        question=question,
        context="\n\n---\n\n".join(context_parts),
    )
    return {
        "keyword": keyword,
        "answer": answer,
        "sources": sources,
}