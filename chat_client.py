"""
Local REPL for the deployed TTTInference class.

`modal run` swallows stdin, so the interactive chat loop has to live in a
plain local Python process. Deploy the app once, then run this script:

    modal deploy infer_modal.py
    python chat_client.py --ckpt step_232

Sampling defaults follow Qwen3-8B's recommended thinking-mode setup
(temp=0.6, top_p=0.95, top_k=20); the Qwen team explicitly warns against
greedy decoding (endless repetition), so don't pass --temperature 0.

Per-turn config (system prompt, enable_thinking, sampling) is passed on
every chat_turn call, so a Modal container that scales down and
respawns can keep serving with no recovery dance. The only thing the
new container can't recover is the prior fast-weight carry -- that
died with the old container.
"""

import argparse

import modal

APP_NAME = "inplace-ttt-infer"
CLASS_NAME = "TTTInference"


def _reset(engine, args):
    return engine.chat_reset.remote(
        evolve=args.evolve,
        from_snapshot_name=args.from_snapshot,
    )


def _print_ready(info, args):
    print(f"chat ready  (evolve={info['evolve']}, "
          f"seeded_from_snapshot={info['seeded_from_snapshot']}, "
          f"system_prompt={bool(args.system.strip())}, "
          f"enable_thinking={args.enable_thinking})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--evolve", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--enable-thinking",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Qwen3 thinking mode. Default OFF here because "
                        "our LoRA/TTT was trained on raw papers, never "
                        "on <think>...</think> traces -- thinking tokens "
                        "would be out-of-distribution for both the LoRA "
                        "and the carry. Pass --enable-thinking to opt in.")
    p.add_argument("--system", default="")
    p.add_argument("--from-snapshot", default="")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None,
                   help="default: 0.6 thinking / 0.7 non-thinking")
    p.add_argument("--top-p", type=float, default=None,
                   help="default: 0.95 thinking / 0.8 non-thinking")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--debug", action="store_true",
                   help="print raw tokens, stop reason, special-token "
                        "decode, and thinking trace for each turn")
    args = p.parse_args()

    # Sampling defaults per Qwen3-8B model card.
    if args.temperature is None:
        args.temperature = 0.6 if args.enable_thinking else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8

    Cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    engine = Cls(ckpt=args.ckpt)

    info = _reset(engine, args)
    _print_ready(info, args)
    print("commands: /save <name>, /reset, /quit")
    print()

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # Strip any accidental "you> " or "bot> " prefixes from pasted
        # content -- they're the REPL's own scaffolding and the model
        # has no business seeing them as user input. Loop in case the
        # paste duplicated the prefix.
        while user.startswith(("you>", "bot>")):
            user = user[4:].lstrip()
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            _reset(engine, args)
            print("[reset]\n")
            continue
        if user.startswith("/save"):
            parts = user.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("[usage: /save <name>]\n")
                continue
            try:
                path = engine.save_session.remote(name=parts[1].strip())
                print(f"[saved fast weights -> {path}]\n")
            except Exception as e:
                print(f"[save error: {e}]\n")
            continue

        try:
            reply = engine.chat_turn.remote(
                user_message=user,
                system_prompt=args.system,
                enable_thinking=args.enable_thinking,
                evolve=args.evolve,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"[error: {e}]\n")
            continue

        print(f"bot> {reply['text']}")
        # Above the --debug gate because these two numbers together tell
        # you whether TTT carry is engaged:
        #   state_ratio grows turn-to-turn => carry is accumulating
        #   state_ratio == 0 but pending climbing toward chunk_size =>
        #     mechanism is alive, just hasn't committed a chunk yet
        #   state_ratio == 0 AND pending flat => carry is actually dead
        print(f"  [state_ratio={reply['state_ratio_mean']:.3e}  "
              f"pending={reply['pending_tokens']}/{reply['chunk_size']}]")
        if args.debug:
            if reply.get("thinking_text"):
                print(f"  [thinking: {reply['thinking_text']!r}]")
            print(f"  [stop_reason={reply['stop_reason']} "
                  f"stop_token_id={reply['stop_token_id']} "
                  f"n_tokens={len(reply['token_ids'])}]")
            print(f"  [token_ids={reply['token_ids']}]")
            print(f"  [raw={reply['raw']!r}]")
        print()


if __name__ == "__main__":
    main()
