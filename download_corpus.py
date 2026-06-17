"""Download a slice of C4 (allenai/c4) into data/corpus.txt.

Usage
-----
    python download_corpus.py              # default 500k lines
    python download_corpus.py --lines 100000
    python download_corpus.py --lines 1000000

C4 is streamed — no need to download the full 300 GB.
Only the requested number of lines is fetched and saved.
Each line is one sentence or short paragraph, already cleaned.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lines', type=int, default=500_000,
                        help='Number of lines to download (default 500000)')
    parser.add_argument('--out', default='data/corpus.txt',
                        help='Output file (default data/corpus.txt)')
    parser.add_argument('--lang', default='en',
                        help='C4 language subset (default en)')
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print('pip install datasets')
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'>>> Streaming C4 ({args.lang}) -> {out}')
    print(f'>>> Target: {args.lines:,} lines  (Ctrl+C to stop early)\n')

    ds = load_dataset(
        'allenai/c4',
        args.lang,
        split='train',
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    skipped = 0
    with open(out, 'w', encoding='utf-8') as f:
        for row in ds:
            text = row.get('text', '').strip()
            if not text:
                skipped += 1
                continue

            # Split into sentences on period/newline so each line is short.
            # STDP benefits from short lines — one concept per line.
            for line in text.replace('\n', ' ').split('. '):
                line = line.strip()
                if len(line) < 10:   # skip fragments
                    continue
                if not line.endswith('.'):
                    line += '.'
                f.write(line + '\n')
                count += 1
                if count >= args.lines:
                    break

            if count % 10_000 == 0 and count > 0:
                print(f'  {count:,} / {args.lines:,} lines written...', flush=True)

            if count >= args.lines:
                break

    print(f'\n>>> Done. {count:,} lines written to {out}')
    print(f'>>> Skipped {skipped:,} empty rows')
    print(f'>>> File size: {out.stat().st_size / 1024 / 1024:.1f} MB')
    print(f'\n>>> Run training:')
    print(f'>>>   python main_agent.py')


if __name__ == '__main__':
    main()
