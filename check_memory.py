"""check_memory.py — estimate VRAM/RAM requirements before training.

Usage
-----
    python check_memory.py
    python check_memory.py --neurons 500000 --density 0.001
"""

import argparse
import math


def fmt(gb: float) -> str:
    if gb < 1.0:
        return f'{gb*1024:.0f} MB'
    return f'{gb:.2f} GB'


def estimate(num_neurons: int, density: float):
    # Neurogenesis: starts at 20% active, grows to 80%
    active_start = int(num_neurons * 0.20)
    active_peak  = int(num_neurons * 0.80)

    # Synapses at start and at full maturity
    syn_start = int(active_start ** 2 * density)
    syn_peak  = int(active_peak  ** 2 * density)

    # Bytes per synapse: 2×int64 indices + float32 weight + float32 integrity
    #                    + float32 stp_u + float32 stp_x = 32 bytes
    BYTES_PER_SYN   = 32
    # Bytes per neuron: pre_trace + post_trace + adaptation (float32×3)
    #                   + refractory (int16) + is_inhibitory (bool)
    #                   + activity vector (float32) ≈ 17 bytes → round to 20
    BYTES_PER_NEURON = 20

    neuron_mem_gb   = num_neurons * BYTES_PER_NEURON / 1e9
    syn_start_gb    = syn_start   * BYTES_PER_SYN   / 1e9
    syn_peak_gb     = syn_peak    * BYTES_PER_SYN   / 1e9

    # Sensory layout (same fractions as main_agent.py)
    vision_size  = (round(num_neurons * 0.00576) // 5) * 5
    audio_size   = max(64,  int(num_neurons * 0.00154))
    hunger_size  = int(num_neurons * 0.005)
    pain_size    = int(num_neurons * 0.010)
    token_size   = int(num_neurons * 0.00819) * 2
    motor_size   = int(num_neurons * 0.015)

    synapses_per_neuron_start = int(active_start * density)
    synapses_per_neuron_peak  = int(active_peak  * density)

    total_start_gb = neuron_mem_gb + syn_start_gb
    total_peak_gb  = neuron_mem_gb + syn_peak_gb

    # GPU recommendations
    gpus = [
        ('GTX 1050 Ti',  4),
        ('GTX 1060 6GB', 6),
        ('RTX 3060',     12),
        ('RTX 4070 Ti',  12),
        ('RTX 3090',     24),
        ('Tesla T4',     16),
        ('A100',         40),
    ]

    print()
    print('=' * 54)
    print(f'  Нейронов:   {num_neurons:>12,}')
    print(f'  Плотность:  {density:>12.6f}')
    print('=' * 54)

    print()
    print('  Сенсорные зоны (нейронов):')
    print(f'    Зрение   (FovealRetina):  {vision_size:>8,}')
    print(f'    Слух     (Cochlea ERB):   {audio_size:>8,}')
    print(f'    Голод    (somatosensory): {hunger_size:>8,}')
    print(f'    Боль     (somatosensory): {pain_size:>8,}')
    print(f'    Токены   (language):      {token_size:>8,}')
    print(f'    Мотор    (motor cortex):  {motor_size:>8,}')

    print()
    print('  Нейрогенез:')
    print(f'    Старт (20% активных):  {active_start:>10,} нейронов')
    print(f'    Пик   (80% активных):  {active_peak:>10,} нейронов')

    print()
    print('  Синапсов:')
    print(f'    На старте:  {syn_start:>14,}  ({synapses_per_neuron_start} / нейрон)')
    print(f'    На пике:    {syn_peak:>14,}  ({synapses_per_neuron_peak} / нейрон)')

    stdp_ok_start = synapses_per_neuron_start >= 100
    stdp_ok_peak  = synapses_per_neuron_peak  >= 200
    print(f'    STDP на старте: {"✓ достаточно" if stdp_ok_start else "✗ мало (<100/нейрон)"}')
    print(f'    STDP на пике:   {"✓ достаточно" if stdp_ok_peak  else "✗ мало (<200/нейрон)"}')

    print()
    print('  Память:')
    print(f'    Нейроны (тензоры):  {fmt(neuron_mem_gb):>10}')
    print(f'    Синапсы на старте:  {fmt(syn_start_gb):>10}')
    print(f'    Синапсы на пике:    {fmt(syn_peak_gb):>10}')
    print(f'    Итого на старте:    {fmt(total_start_gb):>10}')
    print(f'    Итого на пике:      {fmt(total_peak_gb):>10}')
    print('    * Это optimistic lower-bound estimate: не учитывает все временные')
    print('      тензоры и peak memory при инициализации графа/кэшей на GPU.')

    print()
    print('  Совместимость GPU:')
    for name, vram_gb in gpus:
        fits_start = total_start_gb <= vram_gb * 0.85
        fits_peak  = total_peak_gb  <= vram_gb * 0.85
        if fits_peak:
            status = '✓ влезает полностью'
        elif fits_start:
            status = '~ влезает на старте, OOM при росте нейрогенеза'
        else:
            status = '✗ не влезает'
        print(f'    {name:<16} {vram_gb:>2} GB   {status}')

    print()
    if not stdp_ok_peak:
        print('  ⚠  Мало синапсов на нейрон — STDP будет работать плохо.')
        print('     Увеличь density или уменьши num_neurons.')
    if total_peak_gb > 12:
        print('  ⚠  Для полного нейрогенеза нужна карта с большим VRAM.')
        print('     Или снизь num_neurons / density.')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neurons', type=int,   default=None)
    parser.add_argument('--density', type=float, default=None)
    args = parser.parse_args()

    if args.neurons and args.density:
        estimate(args.neurons, args.density)
        return

    # Interactive mode
    print('\n  MindAI — калькулятор памяти')
    print('  Оставь пустым для значений по умолчанию\n')

    try:
        n = input('  Нейронов   [2000000]: ').strip()
        d = input('  Плотность  [0.0002]:  ').strip()
        num_neurons = int(n)   if n else 2_000_000
        density     = float(d) if d else 0.0002
    except (ValueError, KeyboardInterrupt):
        print('Неверный ввод.')
        return

    estimate(num_neurons, density)


if __name__ == '__main__':
    main()
