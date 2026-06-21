from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Ticket


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