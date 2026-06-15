import httpx

async def reason_over_hazard(reasoning_url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f'{reasoning_url}/reason', json=payload)
        response.raise_for_status()
        return response.json()
