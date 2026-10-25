from typing import ClassVar

from juno.domain.session import Session
from juno.domain.values import Message, Role
from juno.ports import LlmClient
from juno.service.journal import Journal


class Compactor:
    ASK: ClassVar[str] = (
        "Write a summary of this conversation so a fresh session can carry on without it. "
        "Cover the user's goal, decisions made, files read or changed and what each change "
        "was, commands that mattered and their results, and what is still to do. Be dense "
        "and concrete; use file paths and names. Reply with the summary only, as text, "
        "without calling tools."
    )

    def __init__(self, llm: LlmClient, journal: Journal) -> None:
        self._llm = llm
        self._journal = journal

    async def summarize(self, session: Session, *, focus: str = "") -> str:
        if not session.messages:
            raise ValueError("there is nothing to compact yet")

        ask = f"{self.ASK}\n\nFocus on: {focus}" if focus else self.ASK
        request = self._journal.request([*session.messages, Message(role=Role.USER, content=ask)])
        reply = await self._llm.complete(request)
        session.usage = session.usage + reply.usage
        await self._journal.save(session)
        text = "\n\n".join(
            str(block.get("text", "")) for block in reply.content if block.get("type") == "text"
        ).strip()

        if not text:
            raise ValueError("the model returned no summary")

        return text
