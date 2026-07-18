from neuromorphic.server.__main__ import build_parser


def test_parser_defaults_and_overrides():
    args = build_parser().parse_args(["--trace", "outputs/live.jsonl"])
    assert args.trace == "outputs/live.jsonl"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.poll == 0.15

    args = build_parser().parse_args(["--trace", "x.jsonl", "--port", "9001", "--poll", "0.05"])
    assert args.port == 9001 and args.poll == 0.05
