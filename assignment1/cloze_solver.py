from typing import List
import re
import os
import pickle
from collections import defaultdict
from collections import Counter

class ClozeSolver:
    def __init__(self,
                 input_filename: str,
                 candidates_filename: str,
                 corpus_filename: str,
                 left_only: bool):

        self.left_only = left_only
        self.candidates_filename = candidates_filename
        self.corpus_filename = corpus_filename
        self.candidates_words = self._get_candidates_words()
        self.words_before, self.words_after = self._get_target_words(input_filename)
        self.bigram_filename = 'bigram.pkl'
        self.unigram_counts = None
        self.bigram_counts = defaultdict(Counter)
        self.bigram_probabilities = defaultdict(lambda: defaultdict(float))

    def _get_candidates_words(self) -> List[str]:
        with open(self.candidates_filename, 'r', encoding='utf-8') as candidates_file:
            return candidates_file.read().splitlines()

    def _get_target_words(self, input_filename):
        words_before = []
        words_after = []

        with open(input_filename, 'r', encoding='utf-8') as fin:
            text = fin.read()

        # clean the text of punctuation except for spaces and dashes
        text_cleaned = re.sub(r'[^\w\s-]', '', text)

        # find words before and after each "__________"
        matches = re.findall(r'(\b\w+\b)\s+__________\s+(\b\w+\b)', text_cleaned)

        # separate words before and after
        words_before.extend([pair[0] for pair in matches])
        words_after.extend([pair[1] for pair in matches])

        return words_before, words_after

    def solve_cloze(self) -> List[str]:
        return list()

    def _load_bigram(self):
        if self.bigram_probabilities and self.bigram_counts:
            return
        if not os.path.isfile(self.bigram_filename):
            self._init_bigram()
            print("\nsaving bigram to file ...")
            pickle.dump(self.bigram_probabilities, open(self.bigram_filename, 'wb'))
            print("finished saving bigram to file ...")
        else:
            print("\nloading bigram pkl ...")
            self.bigram_probabilities = pickle.load(open(self.bigram_filename, 'rb'))
            print("loaded bigram pkl ...")

    def _init_bigram(self) -> None:
        self._init_bigram_counts()
        self._init_bigram_probabilities()

    def _init_bigram_counts(self) -> None:
        words_before_set = set(self.words_before)
        words_after_set = set(self.words_after)
        candidates_words_set = set(self.candidates_words)

        # Regular expression to remove punctuation
        punctuation_re = re.compile(r'[^\w\s-]')

        with open(self.corpus_filename, 'r', encoding='utf-8') as fin:
            print('creating bigram from corpus ...')
            for i, line in enumerate(fin):
                clean_line = punctuation_re.sub('', line).lower()
                words = clean_line.split()

                if len(words) < 2:
                    continue

                # Count bigrams once
                pair_counts = Counter(zip(words, words[1:]))

                # Loop only over bigrams that actually appear
                for (w1, w2), count in pair_counts.items():
                    # Pattern 1: before → candidate
                    if w1 in words_before_set and w2 in candidates_words_set:
                        self.bigram_counts[w1][w2] += count
                    # Pattern 2: candidate → after
                    elif not self.left_only and w1 in candidates_words_set and w2 in words_after_set:
                        self.bigram_counts[w1][w2] += count

                if i % 100000 == 0:
                    print(f"Finished {i} lines...")

    def _init_bigram_probabilities(self) -> None:
        for word1, inner_dict in self.bigram_counts.items():
            total_count = sum(inner_dict.values())  # Calculate total count for the inner dictionary
            for word2, count in inner_dict.items():
                self.bigram_probabilities[word1][word2] = count / total_count  # Compute and store frequency