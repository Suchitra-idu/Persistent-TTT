"""The carry-ablation REPL: talk to the model with each mechanism on and off.

    python -m ttt.experiments.chat_repl --resume-from step_600 --seed-source RedPajamaArXiv

`modal run` swallows stdin, so this is a plain local process against a deployed
inference class. Commands: /m /ab /switch /diag /reset /quit.

Expect some incoherence — the base is Qwen3 continual-pretrained on raw
SlimPajama, never instruction-tuned on chat traces. That is what `--context full`
and `/ab` are for: establish the model is not simply broken before blaming the carry.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Sequence

from ttt.app import chat
from ttt.app.chat import CONTEXTS, FULL, NONE, ChatSession, Sampling, Switches

QUIT = "quit"
RESET = "reset"
MULTILINE = "multiline"
AB = "ab"
SWITCH = "switch"
DIAG = "diag"
SAY = "say"

_PASTE_PREFIXES = ("you>", "bot>")
_TOGGLES = {"within_turn", "cross_turn"}


@dataclass(frozen=True)
class Command:
    kind: str
    text: str = ""


def parse(line: str) -> Command:
    """A bare line is a turn; a leading slash is a command. Pasting a previous
    transcript back in is common enough that the prompts are stripped."""
    line = line.strip()
    while line.startswith(_PASTE_PREFIXES):
        line = line[4:].lstrip()
    if not line.startswith("/"):
        return Command(SAY, line)

    verb, _, rest = line[1:].partition(" ")
    known = {
        "q": QUIT, "quit": QUIT,
        "r": RESET, "reset": RESET,
        "m": MULTILINE,
        "ab": AB,
        "switch": SWITCH,
        "diag": DIAG,
    }
    if verb not in known:
        raise ValueError(f"unknown command /{verb}; try /m /ab /switch /diag /reset /quit")
    return Command(known[verb], rest.strip())


def apply_switch(switches: Switches, argument: str) -> Switches:
    """`/switch cross_turn off`, `/switch context full`, `/switch seed <source>`."""
    name, _, value = argument.partition(" ")
    name, value = name.strip(), value.strip()
    if name in _TOGGLES:
        return dataclasses.replace(switches, **{name: _flag(name, value)})
    if name == "context":
        if value not in CONTEXTS:
            raise ValueError(f"context must be one of {list(CONTEXTS)}, got {value!r}")
        return dataclasses.replace(switches, context=value)
    if name == "seed":
        return dataclasses.replace(switches, seed=value)
    raise ValueError(
        f"unknown switch {name!r}; expected one of "
        f"{sorted(_TOGGLES | {'context', 'seed'})}"
    )


def _flag(name: str, value: str) -> bool:
    if value in ("on", "true", "1"):
        return True
    if value in ("off", "false", "0"):
        return False
    raise ValueError(f"{name} takes on or off, got {value!r}")


def format_diagnostics(turn: chat.Turn) -> str:
    """The four numbers that say whether the carry is doing anything."""
    d = turn.diagnostics
    gate = "" if d.gate_mean is None else f"  gate {d.gate_mean:.3f}±{d.gate_std:.3f}"
    return (
        f"  [state/W0 {d.state_ratio:.3e}  +{d.state_growth:.1%} this turn  "
        f"pending {d.pending_tokens}/{d.chunk_size}{gate}]"
    )


def format_switches(switches: Switches) -> str:
    return (
        f"within_turn={switches.within_turn} cross_turn={switches.cross_turn} "
        f"seed={switches.seed or '(none)'} context={switches.context}"
    )


def ab_lines(message: str, session, control: Switches) -> list[str]:
    """The same turn under two switch settings, sequentially.

    One model means one fast weight, so the arms cannot run side by side the way
    `chat.ab_turn` does — each is reset first instead, and the shared
    `sampling.seed` is what makes a difference attributable to the switches.
    """
    original = session.switches
    lines: list[str] = []
    try:
        for switches in (original, control):
            session.switches = switches
            session.reset()
            turn = session.turn(message)
            lines.append(f"[{format_switches(switches)}]")
            lines.append(turn.text)
            lines.append(format_diagnostics(turn))
    finally:
        session.switches = original
        session.reset()
    return lines


def control_for(switches: Switches) -> Switches:
    """The carry switched off: what the same prompt reads like with no memory."""
    return dataclasses.replace(switches, within_turn=False, cross_turn=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chat_repl")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--flags", default="", help="name=value,name=value")
    parser.add_argument("--system", default="")
    parser.add_argument("--seed-source", default="", help="install this source's carrier")
    parser.add_argument("--context", choices=CONTEXTS, default=NONE)
    parser.add_argument(
        "--within-turn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cross-turn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=False,
        help="Qwen3 thinking mode. Off: the LoRA and the carry were trained on "
             "raw text, never on <think> traces",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--sampling-seed", type=int, default=None,
        help="fixes the draw so /ab attributes differences to the switches",
    )
    return parser


def switches_from(args) -> Switches:
    return Switches(
        within_turn=args.within_turn,
        cross_turn=args.cross_turn,
        seed=args.seed_source,
        context=args.context,
    )


def sampling_from(args) -> Sampling:
    return Sampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.sampling_seed,
    )


def repl(session: ChatSession, *, read=input, write=print) -> None:
    """The loop. Everything it decides is in `parse` and `apply_switch`, which
    is why those are pure and this is not tested."""
    write(f"chat ready  [{format_switches(session.switches)}]")
    write("commands: /m /ab <msg> /switch <name> <value> /diag /reset /quit")
    while True:
        try:
            command = parse(read("you> "))
        except (EOFError, KeyboardInterrupt):
            write("")
            return
        except ValueError as error:
            write(f"[{error}]")
            continue

        if command.kind == QUIT:
            return
        if command.kind == RESET:
            session.reset()
            write("[reset]")
            continue
        if command.kind == DIAG:
            write(f"[{format_switches(session.switches)}]")
            continue
        if command.kind == SWITCH:
            try:
                session.switches = apply_switch(session.switches, command.text)
            except ValueError as error:
                write(f"[{error}]")
                continue
            session.reset()
            write(f"[{format_switches(session.switches)}]  (session reset)")
            continue
        if command.kind == MULTILINE:
            command = Command(SAY, _read_block(read, write))
        if command.kind == AB:
            for line in ab_lines(
                command.text, session, control_for(session.switches)
            ):
                write(line)
            continue
        if not command.text:
            continue

        turn = session.turn(command.text)
        write(f"bot> {turn.text}")
        write(format_diagnostics(turn))


def _read_block(read, write) -> str:
    write("[multiline: end with an empty line]")
    lines = []
    while True:
        try:
            line = read("... ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


class RemoteSession:
    """A ChatSession living in a container, with the surface `repl` needs.

    Switches are pushed on assignment rather than per turn, because the seed
    switch only takes effect at a reset — installing a carrier mid-turn would
    silently mean something different from installing it at the start.
    """

    def __init__(self, handle, *, switches: Switches, sampling: Sampling, system: str):
        self._handle = handle
        self._sampling = sampling
        self._system = system
        self.switches = switches

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name == "switches":
            self.reset()

    def reset(self) -> None:
        self._handle.configure.remote(
            switches=dataclasses.asdict(self.switches),
            sampling=dataclasses.asdict(self._sampling),
            system_prompt=self._system,
        )

    def turn(self, message: str) -> chat.Turn:
        payload = self._handle.turn.remote(message=message)
        return chat.Turn(
            **{**payload, "diagnostics": chat.Diagnostics(**payload["diagnostics"])}
        )


def main(argv: Sequence[str] | None = None) -> None:
    import modal

    args = build_parser().parse_args(argv)
    handle = modal.Cls.from_name("ttt-chat-v1", "ChatEngine")(
        resume_from=args.resume_from, flags=args.flags
    )
    repl(
        RemoteSession(
            handle,
            switches=switches_from(args),
            sampling=sampling_from(args),
            system=args.system,
        )
    )


if __name__ == "__main__":
    main()
