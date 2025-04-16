import numpy as np
import matplotlib.pyplot as plt


def manchester(data):
    return [(0, 1) if bit == '0' else (1, 0) for bit in data]

def differential_manchester(data):
    encoded = []
    last_state = 0

    for bit in data:
        if bit == '0':
            encoded.append((last_state, last_state ^ 1))
        else:
            last_state ^= 1
            encoded.append((last_state, last_state ^ 1))

    return encoded

def ami(data):
    encoded = []
    last_mark = -1

    for bit in data:
        if bit == '0':
            encoded.append(0)
        else:
            last_mark *= -1
            encoded.append(last_mark)

    return encoded

def mlt3(data):
    levels = [0]
    current_level = 0
    direction = 1

    for bit in data:
        bit = int(bit)
        if bit == 1:
            if current_level == 0:
                current_level = direction
            elif current_level == 1:
                current_level = 0
                direction = -1
            elif current_level == -1:
                current_level = 0
                direction = 1
        levels.append(current_level)

    return levels[1:]

def pam5(data):
    mapping = {"00": -2, "01": -1, "10": 1, "11": 2}
    data = data + "0" * (len(data) % 2)
    symbols = [mapping[data[i:i + 2]] for i in range(0, len(data), 2)]
    return symbols

def e_4b5b(input_bits):
    encoding_table = {
        "0000": "11110",
        "0001": "01001",
        "0010": "10100",
        "0011": "10101",
        "0100": "01010",
        "0101": "01011",
        "0110": "01110",
        "0111": "01111",
        "1000": "10010",
        "1001": "10011",
        "1010": "10110",
        "1011": "10111",
        "1100": "11010",
        "1101": "11011",
        "1110": "11100",
        "1111": "11101"
    }

    encoded_bits = ""
    for i in range(0, len(input_bits), 4):
        block = input_bits[i:i + 4]

        if len(block) < 4:
            block = block.ljust(4, '0')

        encoded_bits += encoding_table[block]

    print("4b/5b bin:", encoded_bits)
    print("4b/5b hex:", hex(int(encoded_bits, 2))[2:].upper())
    print("message length (bit/byte):", len(encoded_bits), "/", len(encoded_bits)/8)
    print("redundancy:",  (len(encoded_bits) - len(input_bits)) / len(input_bits))

def scramble(sequence):
    sequence = [int(bit) for bit in sequence]
    scrambled_sequence = []

    for i in range(len(sequence)):
        res = 0
        for offset in [1, 2, 3, 7]:
            if i - offset >= 0:
                res ^= scrambled_sequence[i - offset]

        scrambled_sequence.append(sequence[i] ^ res)

    scrambled_str = ''.join(map(str, scrambled_sequence))

    print("scrambled bin:", scrambled_str)
    print("scrambled hex:", hex(int(scrambled_str, 2))[2:].upper())


def plot_signal_manchester(data, title):
    t = np.arange(0, len(data) * 2, 1)
    signal = np.array(data).flatten()

    plt.step(t, signal, where='post', color='b', linewidth=2)
    plt.ylim(-0.5, 1.5)
    plt.xlim(0, len(signal))
    plt.yticks([0, 1], ['0', '1'])
    plt.xticks(range(0, len(signal), 2), list(binary_data))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel("Бит")
    plt.ylabel("Уровень сигнала")
    plt.show()

def plot_signal_ami(data, title):
    t = np.arange(0, len(data) * 2, 1)
    signal = np.repeat(data, 2)

    plt.step(t, signal, where='post', color='b', linewidth=2)
    plt.ylim(-1.5, 1.5)
    plt.xlim(0, len(signal))
    plt.yticks([-1, 0, 1], ['-1', '0', '+1'])
    plt.xticks(range(0, len(signal), 2), list(binary_data))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel("Бит")
    plt.ylabel("Уровень сигнала")
    plt.show()

def plot_signal_mlt3(data, title):
    t = np.arange(len(data) * 2)
    signal = np.repeat(data, 2)

    plt.step(t, signal, where='post', color='b', linewidth=2)
    plt.ylim(-1.5, 1.5)
    plt.xlim(0, len(signal))
    plt.yticks([-1, 0, 1], ['-1', '0', '+1'])
    plt.xticks(range(0, len(signal), 2), list(binary_data))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel("Бит")
    plt.ylabel("Уровень сигнала")
    plt.show()

def plot_signal_pam5(data, title):
    t = np.arange(len(data) * 2)
    signal = np.repeat(data, 2)

    plt.step(t, signal, where='post', color='b', linewidth=2)
    plt.ylim(-3, 3)
    plt.xlim(0, len(signal))
    plt.yticks([-2, -1, 0, 1, 2], ['-2', '-1', '0', '+1', '+2'])
    plt.xticks(range(0, len(signal)), list(binary_data + "0" * (len(binary_data) % 2)))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel("Бит")
    plt.ylabel("Уровень сигнала")
    plt.show()


binary_data = ""

encoded_thomas = manchester(binary_data)
plot_signal_manchester(encoded_thomas, "Манчестерское кодирование")

diff_encoded = differential_manchester(binary_data)
plot_signal_manchester(diff_encoded, "Дифференциальное манчестерское кодирование")

encoded_ami = ami(binary_data)
plot_signal_ami(encoded_ami, "Биполярный код AMI")

encoded_mlt3 = mlt3(binary_data)
plot_signal_mlt3(encoded_mlt3, "Кодирование MLT-3")

encoded_pam5 = pam5(binary_data)
plot_signal_pam5(encoded_pam5, "Кодирование PAM-5")

encoded_bits = e_4b5b(binary_data)

scrambled = scramble(binary_data)


### ----------------------------------------- ###
def calculate_average_frequency_manchester(bit_sequence: str, f0: float) -> float:
    frequencies = []
    i = 0
    n = len(bit_sequence)

    while i < n:
        current = bit_sequence[i]
        length = 1
        while i + 1 < n and bit_sequence[i + 1] == current:
            i += 1
            length += 1

        freq = f0 / length
        frequencies.extend([freq]*length)
        i += 1

    return sum(frequencies)/ len(bit_sequence), frequencies


bit_seq = ""
f0 = 1000
f_avg, listt = calculate_average_frequency_manchester(bit_seq, f0)
element_counts = {}
for element in listt:
    if element in element_counts:
        element_counts[element] += 1
    else:
        element_counts[element] = 1

for element, count in element_counts.items():
    print(f"Элемент {element} встречается {count} раз(а)")

print(len(bit_seq))
print(f"Средняя частота для последовательности составляет ≈ {f_avg:.2f} Гц")
