
import httpx
import os
from dotenv import load_dotenv
from logging_config import logger
load_dotenv()


LLM_ROUTER_URL = os.getenv("LLM_ROUTER_URL","http://localhost:7001/invoke")


async def get_llm(prompt: str, purpose: str,trace_id:str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            # #print("LLM_ROUTER_URL:", LLM_ROUTER_URL)
            response = await client.post(
                LLM_ROUTER_URL,
                json={
                    "purpose": purpose,
                    "prompt": prompt,
                    "trace_id":trace_id
                },
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()["response"]
    except Exception as e:
        logger.error(f"Error calling LLM Router: {e}")
        raise