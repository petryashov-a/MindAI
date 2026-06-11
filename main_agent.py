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
    python main_agent.py --gui                    # browser UI (recommended)
    python main_agent.py --remote ws://IP:8000    # brain on remote GPU
    python main_agent.py --download ws://IP:8000  # pull weights
    python main_agent.py --data /other/path

Save: savegame_brain/  (shared with main_minecraft.py)
"""

import sys
import time
from pathlib import Path

from mindai import Brain
from mindai.worlds.agent_world import AgentWorld
from mindai.neurochemistry.neuromodulators import EndocrineSystem

# ---------------------------------------------------------------------------
# Brain configuration — must match main_minecraft.py for weight compatibility
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

_CURRICULUM   = (0.65, 0.30, 0.05)   # text / paired(images+video) / qa

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
    remote = None
    gui    = False
    c4     = False
    i = 0
    while i < len(args):
        a = args[i]
        if   a == '--data'     and i+1 < len(args): data   = args[i+1]; i += 2
        elif a == '--remote'   and i+1 < len(args): remote = args[i+1]; i += 2
        elif a == '--download' and i+1 < len(args): _download_weights(args[i+1]); sys.exit(0)
        elif a == '--gui':                          gui    = True;       i += 1
        elif a == '--c4':                           c4     = True;       i += 1
        elif a in ('-h', '--help'): print(__doc__); sys.exit(0)
        else: i += 1
    return Path(data), remote, gui, c4


def _c4_stream(min_len: int = 200):
    """In-memory stream of C4 English paragraphs (no disk writes)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print('>>> --c4 требует: pip install datasets')
        return
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    for row in ds:
        t = (row.get('text') or '').strip()
        if len(t) >= min_len:
            yield t


def _build_world(sources: dict, c4: bool = False):
    print('\n>>> Источники данных:')
    for k, v in sources.items():
        print(f'    {k:<8} → {v}')
    if c4:
        print(f'    {"c4":<8} → allenai/c4 (en, streaming, in-memory)')
    if not sources and not c4:
        print('    (нет данных — только интерактивный режим)')
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
        curriculum   = _CURRICULUM,
        text_stream  = _c4_stream() if c4 else None,
    )


def _build_brain(world):
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
    )
    brain.attach(EndocrineSystem())
    brain._clock_energy_scale = _CLOCK_ENERGY_SCALE
    return brain


# ---------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------

def run():
    data_dir, remote, gui, c4 = _parse_args()
    sources = _discover_data(data_dir)

    if gui:
        # Hand off to the Web GUI (it builds its own brain + world)
        from webgui.server import main as webgui_main
        sys.argv = [sys.argv[0], '--data', str(data_dir),
                    '--neurons', str(_NUM_NEURONS)]
        webgui_main()
        return

    if remote:
        _run_remote(sources, remote, c4)
        return

    world = _build_world(sources, c4=c4)
    brain = _build_brain(world)

    if Path(_SAVE_DIR + '/brain.json').exists():
        ans = input('Найден сохранённый мозг. Продолжить? (y/n): ').strip().lower()
        if ans == 'y':
            brain.load(_SAVE_DIR)

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
    print('>>> Запуск. Ctrl+C для сохранения и выхода.\n')
    brain.run(world, headless=True)


# ---------------------------------------------------------------------------
# Remote WebSocket run — brain on friend's GPU, world local
# ---------------------------------------------------------------------------

def _run_remote(sources: dict, server_url: str, c4: bool = False):
    """Run with brain on a remote GPU server.

    Architecture: this process owns the AgentWorld (mic, files, stdin); the
    server runs the FULL brain.run() pipeline. We just answer the brain's
    RPC calls over a single websocket using msgpack-numpy.
    """
    try:
        import websocket    # websocket-client
    except ImportError:
        print('pip install websocket-client')
        return
    try:
        from mindai.worlds.remote_world import serve_world
    except ImportError as e:
        print(f'>>> {e}')
        print('pip install msgpack msgpack-numpy')
        return

    url = server_url.rstrip('/')
    url = url.replace('http://', 'ws://').replace('https://', 'wss://')
    if not url.startswith('ws'):
        url = 'ws://' + url
    if not url.endswith('/ws'):
        url += '/ws'

    print(f'\n>>> Remote WebSocket: {url}')
    print('>>> Сервер исполняет ВЕСЬ brain.run() (PFC, BG, sleep, нейромодуляторы)')
    print('>>> Локально остаётся мир: ретина / cochlea / stdin / файлы\n')

    world = _build_world(sources, c4=c4)

    ws = None
    for attempt in range(10):
        try:
            ws = websocket.create_connection(url, timeout=10,
                                             enable_multithread=True)
            break
        except Exception as e:
            print(f'>>> Попытка {attempt+1}/10: {e}')
            time.sleep(2)
    if ws is None:
        print('>>> Не удалось подключиться.')
        return

    print('>>> Соединение установлено. Ctrl+C — остановить и сохранить.\n')
    try:
        serve_world(ws, world)
    except KeyboardInterrupt:
        print('\n>>> Остановка — сервер сохранит мозг автоматически.')
    finally:
        try:
            ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Download weights from server
# ---------------------------------------------------------------------------

def _download_weights(server_url: str):
    try:
        import requests, zipfile, io
    except ImportError:
        print('pip install requests')
        return

    base = server_url.rstrip('/').replace('ws://', 'http://').replace('wss://', 'https://')
    if base.endswith('/ws'):
        base = base[:-3]

    print('>>> Сохраняем веса на сервере...')
    try:
        requests.post(f'{base}/save', timeout=30)
    except Exception as e:
        print(f'>>> Save error: {e}')

    print(f'>>> Скачиваем savegame_brain.zip ...')
    r = requests.get(f'{base}/weights/download', timeout=300, stream=True)
    if r.status_code != 200:
        print(f'>>> Ошибка {r.status_code}')
        return

    total = int(r.headers.get('content-length', 0))
    buf   = io.BytesIO()
    done  = 0
    for chunk in r.iter_content(256 * 1024):
        buf.write(chunk)
        done += len(chunk)
        if total:
            pct = done / total * 100
            print(f'\r    {done/1024/1024:.1f} / {total/1024/1024:.1f} MB  ({pct:.0f}%)',
                  end='', flush=True)
    print()

    buf.seek(0)
    out = Path(_SAVE_DIR)
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(out)
    print(f'>>> Готово — веса в {out}/')
    print('>>> Запускай: python main_agent.py')


if __name__ == '__main__':
    run()
