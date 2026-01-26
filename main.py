import sys
from tkinter import EventType
from typing import Any

from agent.agent import Agent
from agent.events import AgentEventType
from client.llm_client import LLMClient
import asyncio, click

from ui.renderer import RENDERER, get_console

console = get_console()


class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.renderer = RENDERER(console)

    async def run_single(self, message: str):
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None
        assistant_streaming = False
        final_response: str | None = None

        async for event in self.agent.run(message):
            # print(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.renderer.begin_assistant()
                    assistant_streaming = True
                self.renderer.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_streaming:
                    self.renderer.end_assistant()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error occurred")
                console.print(f"\n[error]Error: {error}[/error]")
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown")
                # tool_kind = None
                tool = self.agent.tool_registry.get(tool_name)
                # if not tool:
                #     tool_kind = None

                # tool_kind = tool.kind.value
                tool_kind = tool.kind.value if tool else None
                self.renderer.tool_call_start(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None):
    cli = CLI()
    # messages = [{"role": "user", "content": prompt}]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)


main()
