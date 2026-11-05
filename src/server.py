"""ASGI application using the official SDK runtime."""

from agenthack import AgentRuntime

from .my_agent import MyAgent

if __name__ == "__main__":
    AgentRuntime(MyAgent()).run()
