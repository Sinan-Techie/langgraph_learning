import json
import os
import uuid
import httpx
from dotenv import load_dotenv

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    SendMessageRequest,
    MessageSendParams,
    Message,
    Part,
    TextPart
)


from utils import get_llm

load_dotenv()

MASTER_NAME = os.getenv("MASTER_AGENT_NAME", "MasterOrchestrator")
SHARED_API_KEY = os.getenv("SHARED_API_KEY", "changeme")
CHILD_AGENTS = os.getenv("CHILD_AGENTS", "http://localhost:9101")


class MasterOrchestratorExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        raw = next((p.root.text for p in context.message.parts if p.root.kind == 'text'), None)
        if not raw:
            return

        try:
            data = json.loads(raw)
        except Exception:
            data = {"payload": raw}

        user_query = data.get("payload", "")

        # ---------------------------------------------------------
        # STEP 1 — fetch agent cards (skip broken agents)
        # ---------------------------------------------------------
        child_urls = [u.strip() for u in CHILD_AGENTS.split(",") if u.strip()]
        agent_cards = {}
        print("child_urls:", child_urls)
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in child_urls:
                try:
                    resolver = A2ACardResolver(client, base_url=url)
                    card = await resolver.get_agent_card()

                    # store ONLY valid agents
                    agent_cards[card.name] = {
                        "card": card,
                        "url": url
                    }

                except Exception:
                    # skip invalid agents
                    continue

        if not agent_cards:
            await event_queue.enqueue_event(new_agent_text_message(
                json.dumps({"error": "No agents available"})
            ))
            return

        # ---------------------------------------------------------
        # STEP 2 — build description block for LLM
        # ---------------------------------------------------------
        description_lines = [
            f"- {name}: {item['card'].description}"
            for name, item in agent_cards.items()
        ]
        description_block = "\n".join(description_lines)

        llm_prompt = f"""
You are an agent router.

USER QUERY:
{user_query}

AVAILABLE AGENTS:
{description_block}

Return JSON only:
{{
  "selected_agent": "<name>"
}}
"""

        llm_reply = await get_llm(llm_prompt, "core", "master")

        try:
            selected_agent = json.loads(llm_reply)["selected_agent"]
        except Exception:
            # fallback to first agent
            selected_agent = next(iter(agent_cards.keys()))

        if selected_agent not in agent_cards:
            selected_agent = next(iter(agent_cards.keys()))

        chosen_card = agent_cards[selected_agent]["card"]
        print("Selected agent:", selected_agent)
        print("Chosen card:", chosen_card)
        # ---------------------------------------------------------
        # STEP 3 — Build valid A2A message
        # ---------------------------------------------------------
        agent_message = Message(
            role='user',
            messageId=str(uuid.uuid4()),
            parts=[
                Part(
                    root=TextPart(
                        kind="text",
                        text=json.dumps({"payload": user_query})
                    )
                )
            ]
        )


        send_req = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=agent_message)
        )
        print("Send Request:", send_req)
        # ---------------------------------------------------------
        # STEP 4 — Call child agent
        # ---------------------------------------------------------
        async with httpx.AsyncClient(timeout=35.0) as client:
            a2a_client = A2AClient(client, chosen_card)
            print("Calling child agent:", selected_agent)
            try:
                child_res = await a2a_client.send_message(send_req)
                print("Child response:", child_res)

                # ---------- Tailored extraction for your exact response shape ----------
                # The response shape we've observed:
                # child_res (wrapper) -> root (SendMessageSuccessResponse) -> result (Message) -> parts -> Part.root.text
                #
                message_obj = None

                # 1) Prefer child_res.result if present (some versions expose result directly)
                if hasattr(child_res, "result") and child_res.result is not None:
                    message_obj = child_res.result

                # 2) Otherwise, check child_res.root.result (wrapper.root.result)
                elif hasattr(child_res, "root") and hasattr(child_res.root, "result") and child_res.root.result is not None:
                    message_obj = child_res.root.result

                # 3) Otherwise, maybe child_res.root itself *is* a Message object
                elif hasattr(child_res, "root") and hasattr(child_res.root, "parts"):
                    message_obj = child_res.root

                # 4) As a last fallback, if child_res itself has 'parts', treat it as the Message
                elif hasattr(child_res, "parts"):
                    message_obj = child_res

                # If we still have no message object, return a helpful error
                if message_obj is None:
                    child_payload = {"error": "Could not find message object in child response"}
                else:
                    # Extract the first text part (old-style Part(root=TextPart(...)))
                    extracted_text = None
                    if getattr(message_obj, "parts", None):
                        first_part = message_obj.parts[0]
                        if hasattr(first_part, "root") and hasattr(first_part.root, "text"):
                            extracted_text = first_part.root.text

                    if not extracted_text:
                        child_payload = {"error": "No text part found in child response"}
                    else:
                        # Parse JSON if possible
                        try:
                            child_payload = json.loads(extracted_text)
                        except Exception:
                            child_payload = {"response_text": extracted_text}

            except Exception as e:
                child_payload = {"error": str(e)}
        # ---------------------------------------------------------
        # STEP 5 — return response
        # ---------------------------------------------------------
        await event_queue.enqueue_event(new_agent_text_message(json.dumps({
            "master": MASTER_NAME,
            "selected_agent": selected_agent,
            "agent_response": child_payload
        })))
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return