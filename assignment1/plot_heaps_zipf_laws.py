import json
import pickle
import os.path
from collections import defaultdict
from matplotlib import pyplot as plt
from math import log
import seaborn as sn
sn.set()


def read_data(filename):
    word2freq = defaultdict(int)

    i = 0
    with open(filename, 'r', encoding='utf-8') as fin:
        print('reading the text file...')
        for i, line in enumerate(fin):
            for word in line.split():
                word2freq[word] += 1
            if i % 100000 == 0:
                print(i)

    total_words = sum(word2freq.values())
    word2nfreq = {w: word2freq[w]/total_words for w in word2freq}

    return word2nfreq


def plot_zipf_law(word2nfreq):
    y = sorted(word2nfreq.values(), reverse=True)
    x = list(range(1, len(y)+1))

    product = [a * b for a, b in zip(x, y)]
    print(product[:1000])  # todo: print and note the roughly constant value

    y = [log(e, 2) for e in y]
    x = [log(e, 2) for e in x]

    plt.plot(x, y)
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.title("Zipf's law")
    plt.show()

def plot_heaps_law(corpus_path):
    """
    Plots Heaps' law: log(V) vs log(N), where:
    N = total number of tokens scanned
    V = size of vocabulary (unique words)

    This function reads the corpus progressively, measures vocab growth,
    and renders a log-log plot illustrating Heaps' Law.
    """
    vocab = set() # a set to store unique words as we scan the corpus
    N_values = [] # total number of tokens scanned
    V_values = [] # size of vocabulary (unique words)
    N = 0 # counter for total number of tokens scanned

    CHECKPOINT = 50000  # take measurements every 50k words

    print("Reading corpus for Heaps' Law...")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                N += 1
                vocab.add(word)

                if N % CHECKPOINT == 0:
                    N_values.append(N)
                    V_values.append(len(vocab))
                    print(f"N={N}, V={len(vocab)}")

    # final measurement
    N_values.append(N)
    V_values.append(len(vocab))

    # Plots Heaps' law: log(V) vs log(N)

    # Convert N (total tokens) and V (unique words) into log scale.
    logN = [log(n, 2) for n in N_values]
    logV = [log(v, 2) for v in V_values]

    plt.plot(logN, logV, marker='o')
    plt.xlabel("log(N) (total words)")
    plt.ylabel("log(V) (vocabulary size)")
    plt.title("Heaps' Law")
    plt.grid(True)
    plt.show()


# Note:
# The Zipf plot uses plt.show(), which blocks execution until the plot window is closed.
# To see the Heaps' Law plot afterwards, close the Zipf plot window first.
# This behavior is expected when using matplotlib.
if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    if not os.path.isfile('word2nfreq.pkl'):
        data = read_data(config['corpus'])
        pickle.dump(data, open('word2nfreq.pkl', 'wb'))

    plot_zipf_law(pickle.load(open('word2nfreq.pkl', 'rb')))
    
    print("Plotting Heaps' Law...")
    plot_heaps_law(config['corpus'])
