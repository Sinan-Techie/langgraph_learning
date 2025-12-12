# agent_domain1/domain1_executor.py  (file named domain1_executor.py)
import json
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from utils import get_llm
import os
from dotenv import load_dotenv

load_dotenv()
AGENT_NAME = os.getenv("DOMAIN1_NAME", "Domain1Agent")

class Domain1Executor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract incoming text part (JSON expected)
        raw = next((part.root.text for part in context.message.parts if part.root.kind == 'text'), None)
        if not raw:
            return

        try:
            data = json.loads(raw)
        except Exception:
            data = {"payload": raw}
        
        payload = data.get("payload", {})
        # Hardcoded prompt logic (simple): instruct LLM to "process" the payload for Domain1
        formatted_prompt = f"Domain1 agent: transform the payload into a short summary. Payload: {json.dumps(payload)}"
        # get_llm returns a deterministic JSON-like string
        llm_response = await get_llm(formatted_prompt, "core", trace_id="domain1")

        # Try convert to json-friendly object
        try:
            parsed = json.loads(llm_response)
        except Exception:
            parsed = {"response_text": llm_response}

        await event_queue.enqueue_event(new_agent_text_message(json.dumps({
            "agent": AGENT_NAME,
            "result": parsed
        })))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported")
