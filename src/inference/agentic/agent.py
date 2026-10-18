"""
Build the Pipeline 3 ReAct agent (LangChain classic) with project tools.
"""

from __future__ import annotations

from typing import List

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from src.core.constants import AGENT_MAX_ITERATIONS
from src.inference.agentic.skills import SKILL_REGISTRY
from src.inference.agentic.tools import get_agent_tools

# Must match the stop sequence create_react_agent binds to the LLM below.
REACT_STOP_SEQUENCE = "\nObservation"


class TruncatingReActOutputParser(ReActSingleInputOutputParser):
    """
    ReAct output parser that truncates at the stop sequence before parsing.

    Some providers (observed with deepseek-v3.2 via OpenRouter) do not honor
    the "stop" sequence bound to the LLM, so the model keeps generating past
    the point where it should hand control back to the executor - hallucinating
    further Thought/Action/Observation turns (with fabricated tool outputs) in
    a single completion. Truncating client-side recovers the intended
    single-step ReAct behavior regardless of whether the provider enforced
    the stop sequence itself.
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        idx = text.find(REACT_STOP_SEQUENCE)
        if idx != -1:
            text = text[:idx]
        return super().parse(text)


def build_agent_system_prompt() -> str:
    """
    Build the Pipeline 3 agent system prompt.

    Defines the clone-detection task and rules (use/load skills and tools)
    and includes a list of available skills from ``SKILL_REGISTRY``. Injected into the
    React prompt via the ``{system_prompt}`` placeholder.

    Returns:
        System prompt text for the agent.
    """
    skill_lines = "\n".join(
        f"  SKILL {info['name']}: {info['description']}"
        for info in sorted(SKILL_REGISTRY.values(), key=lambda x: x["name"])
    )

    return (
        "You are a research assistant performing cross-language "
        "code clone detection between Java and Python.\n\n"
        "BACKGROUND:\n"
        "Two code fragments are clones if they implement the same "
        "functionality and produce the same output for the same input, "
        "regardless of language, syntax, or structure differences.\n\n"
        "RULES - follow these strictly:\n"
        "1. Use your available skills to guide your work.\n"
        "2. Skills are instructions for you to follow — they are not tools "
        "you can call. If you need a skill then load the skill with load_skill tool, then follow its "
        "instructions through your own reasoning.\n"
        "3. You MUST call write_result tool before giving your Final Answer. "
        "Never put your verdict in the Final Answer text — always call "
        "write_result tool first and wait for its confirmation.\n"
        "4. Never fabricate, simulate, or assume tool outputs - always call "
        "the tool and wait for the real Observation before proceeding.\n\n"
        f"Available skills:\n{skill_lines}"
    )


def get_react_prompt_template() -> str:
    """
    Return the base React prompt template string used to build the Pipeline 3 agent prompt.

    This template follows the classic ReAct "Thought/Action/Observation" protocol and is
    intended to be wrapped by :class:`langchain_core.prompts.PromptTemplate` and then
    partially filled with ``system_prompt`` (see ``build_agent_system_prompt()``).

    The returned string contains placeholders that are supplied at runtime by the agent
    framework and by the caller:

    - ``{system_prompt}``: Injected via ``PromptTemplate.partial(...)`` in ``build_react_executor``.
    - ``{tools}``: Rendered tool descriptions injected by ``create_react_agent``.
    - ``{tool_names}``: Comma-separated tool name list injected by ``create_react_agent``.
    - ``{input}``: The user task, supplied to ``AgentExecutor.invoke({"input": ...})``.
    - ``{agent_scratchpad}``: The running internal trace of prior Thought/Action/Observation
      steps for the current agent episode, injected/maintained by the executor.

    Returns:
        A format string compatible with ``PromptTemplate.from_template(...)`` for constructing a React agent prompt.
    """
    return (
        "{system_prompt}"
        "\n\nYou have access to these tools:\n\n{tools}\n\n"
        "Use this format strictly:\n\n"
        "Question: the input task you must solve\n"
        "Thought: plan your next step\n"
        "Action: the action to take, must be one of [{tool_names}]\n"
        "Action Input: a JSON string assigned to `data` or leave blank for zero-arg tools.\n"
        "Observation: tool output\n"
        "... repeat Thought/Action/Action Input/Observation as needed ...\n"
        "Once write_result has been called and its confirmation appears in the Observation,\n"
        "output your final Thought and Final Answer in a new step:\n"
        "Thought: I have completed detection and recorded the result.\n"
        "Final Answer: CLONE or NOT_CLONE\n\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )


def build_react_executor(llm: BaseLanguageModel) -> AgentExecutor:
    """
    Create an AgentExecutor with react prompting and repository tools.

    Args:
        llm: Chat model from :mod:`src.llm`.

    Returns:
        Configured AgentExecutor (verbose, bounded iterations, parsing error tolerance).
    """
    prompt = PromptTemplate.from_template(
        get_react_prompt_template()
    ).partial(system_prompt=build_agent_system_prompt())

    tools: List[BaseTool] = get_agent_tools()
    agent = create_react_agent(
        llm, tools, prompt, output_parser=TruncatingReActOutputParser()
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=AGENT_MAX_ITERATIONS,
        handle_parsing_errors=False,
    )
