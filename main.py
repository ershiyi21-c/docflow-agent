import json
from embedding import embed_text, embed_texts
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Document, DocumentChunk, DocumentChunkEmbedding, Ticket
from llm import generate_knowledge_answer
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
MIN_SEMANTIC_SCORE = 0.55

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

    chunk_objects = []

    for index, chunk_content in enumerate(chunks):
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk_content,
        )
        chunk_objects.append(chunk)
        db.add(chunk)

    db.commit()

    vectors = embed_texts(
        [chunk.content for chunk in chunk_objects]
    )

    for chunk, vector in zip(chunk_objects, vectors):
        embedding = DocumentChunkEmbedding(
            chunk_id=chunk.id,
            vector_json=json.dumps(vector),
        )
        db.add(embedding)

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

    # 1. 将用户问题转换为向量
    query_vector = embed_text(question)

    # 2. 读取所有已保存向量的文档片段
    vector_results = (
        db.query(
            DocumentChunk,
            DocumentChunkEmbedding,
            Document.original_filename,
        )
        .join(
            DocumentChunkEmbedding,
            DocumentChunkEmbedding.chunk_id == DocumentChunk.id,
        )
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .all()
    )

    # 3. 计算问题向量与每个文档片段向量的相似度
    scored_results = []

    for chunk, embedding, original_filename in vector_results:
        chunk_vector = json.loads(embedding.vector_json)

        score = sum(
            a * b
            for a, b in zip(query_vector, chunk_vector)
        )

        scored_results.append(
            (score, chunk, original_filename)
        )

    # 4. 相似度从高到低排序，取前三条
    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_results = scored_results[:3]

    if (
        not top_results
        or top_results[0][0] < MIN_SEMANTIC_SCORE
    ):
        best_score = (
            round(top_results[0][0], 4)
            if top_results
            else None
        )

        return {
            "answer": "知识库中没有足够相关的资料。",
            "best_score": best_score,
            "sources": [],
        }

    # 5. 组装检索到的资料，交给 DeepSeek 回答
    context_parts = []
    sources = []

    for score, chunk, original_filename in top_results:
        context_parts.append(
            f"文档：{original_filename}\n"
            f"内容：{chunk.content}"
        )

        sources.append(
            {
                "document_id": chunk.document_id,
                "document_name": original_filename,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
            }
        )

    answer = generate_knowledge_answer(
        question=question,
        context="\n\n---\n\n".join(context_parts),
    )

    return {
        "answer": answer,
        "sources": sources,
    }

@app.get("/documents/semantic-search")
def semantic_search_documents(
    query: str,
    top_k: int = 3,
    db: Session = Depends(get_db),
):
    question = query.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="查询内容不能为空",
        )

    query_vector = embed_text(question)

    results = (
        db.query(
            DocumentChunk,
            DocumentChunkEmbedding,
            Document.original_filename,
        )
        .join(
            DocumentChunkEmbedding,
            DocumentChunkEmbedding.chunk_id == DocumentChunk.id,
        )
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .all()
    )

    items = []

    for chunk, embedding, original_filename in results:
        chunk_vector = json.loads(embedding.vector_json)

        score = sum(
            a * b
            for a, b in zip(query_vector, chunk_vector)
        )

        items.append(
            {
                "document_id": chunk.document_id,
                "document_name": original_filename,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
                "content": chunk.content,
            }
        )

    items.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "query": question,
        "total": len(items[:top_k]),
        "items": items[:top_k],
    }