"""main_agent.py — multimodal brain agent with curriculum learning.

Data layout (auto-discovered from data/)
    data/corpus.txt   — text corpus            (raw mode,    65%)
    data/qa.txt       — Q&A pairs              (qa mode,      5%)
    data/images/      — *.png/*.jpg [+ .txt]   (paired mode, 30%)
    data/video/       — *.mp4 [+ .txt]
    data/audio/       — *.wav (optional)

    Missing items are skipped silently.

Usage
    python main_agent.py                          # local GPU, stdin chat
    python main_agent.py --data /other/path       # custom data directory
    python main_agent.py --c4                     # stream from allenai/c4

Save: savegame_brain/
"""

import sys
from pathlib import Path

from mindai import Brain
from mindai.worlds.agent_world import AgentWorld
from mindai.neurochemistry.neuromodulators import EndocrineSystem

# ---------------------------------------------------------------------------
# Brain configuration
# ---------------------------------------------------------------------------

_SAVE_DIR           = 'savegame_brain'
_NUM_NEURONS        = 400000
_SYNAPSE_DENSITY    = 0.001
_CLOCK_ENERGY_SCALE = 0.05

# Sensory channel sizes — all relative to _NUM_NEURONS so the layout scales.
# Vision: 0.576% — FovealRetina (Curcio 1990), must be divisible by 5.
# Audio:  0.154% — A1/V1 ratio in human cortex ~8%/30% × vision (Elston 2003).
#                  Cochlea outputs: 40% sustained / 30% onset / 20% modulation / 10% broadband.
_VISION_SIZE  = (round(_NUM_NEURONS * 0.00576) // 5) * 5   # ~11520 @ 2M
_AUDIO_SIZE   = max(64, int(_NUM_NEURONS * 0.00154))        # ~3080  @ 2M
_HUNGER_SIZE  = int(_NUM_NEURONS * 0.005)                   # ~10000 @ 2M  somatosensory
_PAIN_SIZE    = int(_NUM_NEURONS * 0.010)                   # ~20000 @ 2M  somatosensory
_TOKEN_SIZE   = int(_NUM_NEURONS * 0.00819) * 2             # ~32760 @ 2M  cur+ctx

# Curriculum is strictly sequential/phase-based (see AgentWorld configuration)

# ---------------------------------------------------------------------------

def _discover_data(data_dir: Path) -> dict:
    found: dict[str, str] = {}
    files = {'text': 'corpus.txt', 'qa': 'qa.txt'}
    dirs  = {'images': 'images', 'video': 'video', 'audio': 'audio'}
    for key, name in files.items():
        p = data_dir / name
        if p.exists():
            found[key] = str(p)
    for key, name in dirs.items():
        p = data_dir / name
        if p.is_dir() and any(p.iterdir()):
            found[key] = str(p)
    return found


def _parse_args():
    args   = sys.argv[1:]
    data   = 'data'
    c4     = False
    rehab_ticks = 0
    i = 0
    while i < len(args):
        a = args[i]
        if   a == '--data' and i+1 < len(args): data = args[i+1]; i += 2
        elif a == '--c4':                       c4   = True;       i += 1
        elif a == '--rehab-ticks' and i+1 < len(args): rehab_ticks = int(args[i+1]); i += 2
        elif a in ('-h', '--help'): print(__doc__); sys.exit(0)
        else: i += 1
    return Path(data), c4, rehab_ticks


def _c4_stream(min_len: int = 200):
    """In-memory stream of C4 English paragraphs (no disk writes)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print('>>> --c4 requires: pip install datasets')
        return
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    for row in ds:
        t = (row.get('text') or '').strip()
        if len(t) >= min_len:
            yield t


def _build_world(sources: dict, c4: bool = False):
    print('\n>>> Data sources:')
    for k, v in sources.items():
        print(f'    {k:<8} -> {v}')
    if c4:
        print(f'    {"c4":<8} -> allenai/c4 (en, streaming, in-memory)')
    if not sources and not c4:
        print('    (There are no data sources! Running with empty world...)')
    print()

    return AgentWorld(
        text_corpus  = sources.get('text'),
        images_dir   = sources.get('images'),
        video_dir    = sources.get('video'),
        qa_corpus    = sources.get('qa'),
        audio_source = sources.get('audio'),
        vision_size  = _VISION_SIZE,
        audio_size   = _AUDIO_SIZE,
        interactive  = True,
        text_stream  = _c4_stream() if c4 else None,
    )


def _build_brain(world, rehab_ticks=0):
    sensory = {
        'vision': _VISION_SIZE,
        'audio':  _AUDIO_SIZE,
        'hunger': _HUNGER_SIZE,
        'pain':   _PAIN_SIZE,
        'token':  _TOKEN_SIZE,
    }
    print(f'\n>>> Neurons        : {_NUM_NEURONS:,}')
    print(f'>>> Synapse density: {_SYNAPSE_DENSITY}')
    print(f'>>> Sensory layout : {sensory}\n')

    brain = Brain(
        num_neurons    = _NUM_NEURONS,
        sensory_layout = sensory,
        motor_layout   = world.motor_layout,
        device         = 'auto',
        save_path      = _SAVE_DIR,
        num_actions    = world.tokenizer.vocab_size,
        synapse_density= _SYNAPSE_DENSITY,
        rehab_ticks    = rehab_ticks,
    )
    brain.attach(EndocrineSystem())
    brain._clock_energy_scale = _CLOCK_ENERGY_SCALE
    return brain


# ---------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------

def run():
    data_dir, c4, rehab_ticks = _parse_args()
    sources = _discover_data(data_dir)

    world = _build_world(sources, c4=c4)
    brain = _build_brain(world, rehab_ticks=rehab_ticks)

    if Path(_SAVE_DIR + '/brain.json').exists():
        ans = input('Found saved brain. Continue? (y/n): ').strip().lower()
        if ans == 'y':
            brain.load(_SAVE_DIR)
            world._tick_counter[0] = brain.tick
            world.world_tick = brain.tick
            print(f'>>> Brain successfully restored at tick {brain.tick:,}. Continuing training.')

    tc           = world._tick_counter
    orig_execute = world.execute_action

    def _execute_with_log(motor_idx: int):
        result = orig_execute(motor_idx)
        t = tc[0]
        if t % 1000 == 0 and t > 0:
            print(f'[{t:,}] surprise={brain.surprise:.2f} | mood={brain.mood} '
                  f'| mode={world._mode} | {repr(world.get_current_output()[-60:])}')
        return result

    world.execute_action = _execute_with_log
    print('>>> Starting. Ctrl+C to save and exit.\n')
    brain.run(world, headless=True, checkpoint_interval=50000)


if __name__ == '__main__':
    run()
