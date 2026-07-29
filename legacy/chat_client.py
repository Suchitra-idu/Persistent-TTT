"""Local REPL for the deployed TTTInference class.

`modal run` swallows stdin, so the interactive chat loop runs as a plain
local Python process:

    modal deploy infer_modal.py
    python chat_client.py --ckpt step_232

Sampling defaults follow Qwen3's recommended thinking-mode setup; do not
pass --temperature 0 (Qwen team warns greedy decoding causes endless repetition).
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

    # Sampling defaults per the Qwen3 model card.
    if args.temperature is None:
        args.temperature = 0.6 if args.enable_thinking else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8

    Cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    engine = Cls(ckpt=args.ckpt)

    info = _reset(engine, args)
    _print_ready(info, args)
    print("commands: /m (multiline), /save <name>, /reset, /quit")
    print()

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # /m collects lines until empty, then sends the block as one turn.
        if user == "/m":
            print("[multiline: end with an empty line]")
            lines = []
            while True:
                try:
                    line = input("... ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    break
                lines.append(line)
            user = "\n".join(lines).strip()
            if not user:
                continue
        # Strip accidental "you> "/"bot> " prefixes from pasted content.
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
        # state_ratio + pending together tell you whether TTT carry is engaged.
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
