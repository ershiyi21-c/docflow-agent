from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Ticket

from typing import Literal

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DocFlow Agent",
    description="企业知识库与工单协同 Agent",
    version="0.1.0",
)


class TicketCreate(BaseModel):
    title: str
    content: str
    priority: str = "normal"

class TicketStatusUpdate(BaseModel):
    status: Literal["open", "processing", "closed"]

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