from fastapi import APIRouter
from .approval import requires_approval

router = APIRouter(prefix='/agentic', tags=['agentic-hse'])

@router.post('/reason')
async def reason(payload: dict):
    score = int(payload.get('score', 0))
    return {'ok': True, 'requires_human_approval': requires_approval(score), 'payload': payload}
