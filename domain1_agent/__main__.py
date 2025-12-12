# agent_domain_1/__main__.py
import os
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from domain1_executor import Domain1Executor
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

load_dotenv()

AGENT_NAME = os.getenv("DOMAIN1_NAME", "Domain1Agent")
AGENT_URL = os.getenv("DOMAIN1_URL", "http://0.0.0.0")
AGENT_PORT = int(os.getenv("DOMAIN1_PORT", "9101"))
SHARED_API_KEY = os.getenv("SHARED_API_KEY", "your-secret-key")

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        auth_header = request.headers.get("Authorization")
        if auth_header != f"Bearer {self.api_key}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

if __name__ == "__main__":
    agent_card = AgentCard(
        name=AGENT_NAME,
        description="Domain1 agent (example domain)",
        url=AGENT_URL,
        version="0.1.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Domain1Executor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = server.build()
    app.add_middleware(APIKeyMiddleware, api_key=SHARED_API_KEY)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
