from pathlib import Path
import sys
from tkinter import EventType
from typing import Any

from agent.agent import Agent
from agent.events import AgentEventType
import asyncio, click

from config.config import Config
from config.loader import load_config
from ui.renderer import RENDERER, get_console

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.agent: Agent | None = None
        self.config = config
        self.renderer = RENDERER(config, console)

    def _mask_key(self, key: str | None) -> str:
        if not key:
            return "(not set)"
        k = key.strip().strip('"').strip("'")
        if len(k) <= 8:
            return f"{k[:2]}...{k[-2:]} (len={len(k)})"
        return f"{k[:4]}...{k[-4:]} (len={len(k)})"

    def _print_config(self) -> None:
        console.print(f"model: {self.config.model_name}")
        console.print(f"base_url: {self.config.base_url or '(not set)'}")
        console.print(f"api_key: {self._mask_key(self.config.api_key)}")

    def _print_help(self) -> None:
        console.print("/help - show commands")
        console.print("/config - show active configuration")
        console.print("/model [model_name] - show or set active model")
        console.print("/exit - quit")

    def _handle_command(self, message: str) -> bool:
        msg = message.strip()
        if not msg.startswith("/"):
            return False

        parts = msg.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/help"}:
            self._print_help()
            return True

        if cmd in {"/config"}:
            self._print_config()
            return True

        if cmd in {"/model"}:
            if not arg:
                console.print(f"model: {self.config.model_name}")
                return True
            self.config.model_name = arg
            if self.agent is not None:
                self.agent.context_manager.set_model_name(arg)
            console.print(f"model set to: {self.config.model_name}")
            return True

        return False

    async def run_single(self, message: str):
        async with Agent(self.config) as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(
        self,
    ):
        self.renderer.print_welcome(
            "AI Agent",
            lines=[
                f"model: {self.config.model_name}",
                # f"cwd: {Path.cwd()}",
                f"cwd: {self.config.cwd}",
                "commands: /help /config /approval /model /exit",
            ],
        )
        async with Agent(self.config) as agent:
            self.agent = agent

            while True:
                try:
                    user_input = console.input("\n[user]>[/user] ")
                    user_input = user_input.strip()
                    if not user_input:
                        continue
                    if user_input.lower() in {"/exit", "/quit"}:
                        break
                    if self._handle_command(user_input):
                        continue
                    await self._process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to quit.[/dim]")
                except EOFError:
                    break

        console.print("\n[dim]Goodbye![/dim]")

    def _get_tool_kind(self, tool_name) -> str | None:
        # tool_kind = None
        tool = self.agent.tool_registry.get(tool_name)
        # if not tool:
        # tool_kind = None

        # tool_kind = tool.kind.value
        tool_kind = tool.kind.value if tool else None

        return tool_kind

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
                tool_kind = self._get_tool_kind(tool_name)
                self.renderer.tool_call_start(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                self.renderer.tool_call_complete(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("success", False),
                    event.data.get("output", ""),
                    event.data.get("error"),
                    event.data.get("metadata"),
                    event.data.get("truncated", False),
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Current working directory",
)
def main(
    prompt: str | None,
    cwd: Path | None,
):

    try:
        config = load_config(cwd=cwd)
    except Exception as e:
        console.print(f"[error]Configuration Error: {e}[/error]")

    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[error]{error}[/error]")
        sys.exit(1)

    cli = CLI(config)

    # messages = [{"role": "user", "content": prompt}]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)

    else:
        asyncio.run(cli.run_interactive())


main()
