"""Compare experimental forwarding orchestration against archived revision 03."""
import argparse
import asyncio
from pathlib import Path
from tests import compare_versions as compare
from src import my_agent
from src.forward_service import solve_async


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', nargs='+')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    assert my_agent.solve_async is solve_async, "benchmark must exercise the production entry"
    output = compare.check_output(args.output)
    compare.configure_environment()
    versions = [compare.Version('1.5.3', compare.PROJECT / 'artifacts/v1.5/iterations/03/source/src', False),
                compare.Version('1.5.4', compare.PROJECT / 'src', True)]
    samples = compare.load_samples(args.samples, all_public=args.samples is None)
    return asyncio.run(compare.run(versions, samples, args.repeat, output, compare.apply_limits()))


if __name__ == '__main__':
    raise SystemExit(main())
