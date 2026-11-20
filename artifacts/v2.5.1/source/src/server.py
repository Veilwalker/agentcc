"""ASGI application using the official SDK runtime."""

import gc

from agenthack import AgentRuntime

from .my_agent import MyAgent

if __name__ == "__main__":
    young, middle, old = gc.get_threshold()
    if 0 < young < 24000:
        gc.set_threshold(24000, middle, old)
    AgentRuntime(MyAgent()).run()
