# master_orchestrator/master_orchestrator_executor.py
import json
import os
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
import httpx
from dotenv import load_dotenv

load_dotenv()

# MASTER reads child agent endpoints from env (comma-separated)
CHILD_AGENTS = os.getenv("CHILD_AGENTS", "http://localhost:9101,http://localhost:9102,http://localhost:9103,http://localhost:9104")
SHARED_API_KEY = os.getenv("SHARED_API_KEY", "your-secret-key")
AGENT_NAME = os.getenv("MASTER_AGENT_NAME", "MasterOrchestrator")

class MasterOrchestratorExecutor(AgentExecutor):
    def __init__(self):
        self.child_urls = [u.strip() for u in CHILD_AGENTS.split(",") if u.strip()]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Expected incoming message body (text part) is a JSON string:
        {
          "payload": {...},            <- arbitrary payload
          "route_to": ["domain1","domain3"]  <- optional list; if absent we send to all children
        }
        """
        raw = next((part.root.text for part in context.message.parts if part.root.kind == 'text'), None)
        if not raw:
            # no text -> nothing to do
            return

        try:
            data = json.loads(raw)
        except Exception:
            # consider raw as simple text payload
            data = {"payload": raw}

        payload = data.get("payload", {})
        # optional routing filter by domain names (we will match substrings of child URLs)
        route_to = data.get("route_to", None)  # e.g. ["domain1"]

        responses = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in self.child_urls:
                # simple routing logic: if route_to set, check if any route_to element is substring of url
                if route_to:
                    if not any(rt.lower() in url.lower() for rt in route_to):
                        continue
                try:
                    r = await client.post(
                        url,
                        json={"payload": payload},
                        headers={"Authorization": f"Bearer {SHARED_API_KEY}"},
                        timeout=20.0
                    )
                    r.raise_for_status()
                    try:
                        responses.append({"url": url, "response": r.json()})
                    except Exception:
                        responses.append({"url": url, "response_text": r.text})
                except Exception as exc:
                    responses.append({"url": url, "error": str(exc)})

        # Build aggregated response and send it back to orchestrator caller via event_queue
        aggregated = {"master": AGENT_NAME, "child_responses": responses}
        await event_queue.enqueue_event(new_agent_text_message(json.dumps(aggregated)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Optional: implement cancellation semantics
        raise NotImplementedError("Cancel not implemented for master orchestrator")
