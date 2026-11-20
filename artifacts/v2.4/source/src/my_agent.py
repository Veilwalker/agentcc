"""Official AgentHack SDK adapter; computation lives in the shared service."""

from agenthack import BaseAgent

from .generated_service import solve_async


class MyAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        return await solve_async(task_input)
