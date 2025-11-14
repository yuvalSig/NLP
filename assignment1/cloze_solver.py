from typing import List, Tuple
import re
import os
import pickle
from collections import defaultdict
from collections import Counter
import random

class ClozeSolver:
    def __init__(self,
                 input_filename: str,
                 candidates_filename: str,
                 corpus_filename: str,
                 left_only: bool):

        self.input_filename = input_filename
        self.left_only = left_only
        self.candidates_filename = candidates_filename
        self.corpus_filename = corpus_filename
        self.candidates_words = self._get_candidates_words()
        self.word_before, self.word_after, self.word_before2, self.word_after2 = self._get_target_words()
        # Build n-gram models
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.unigrams = Counter()
        self._init_ngram_counts()

    def _get_candidates_words(self) -> List[str]:
        with open(self.candidates_filename, 'r', encoding='utf-8') as candidates_file:
            return candidates_file.read().splitlines()

    def _get_target_words(self):
        word_before = []  # Immediate word before blank (for bigram: word_before → candidate)
        word_after = []  # Immediate word after blank (for bigram: candidate → word_after)
        word_before2 = []  # Second word before blank (for trigram: word_before2 → word_before → candidate)
        word_after2 = []  # Second word after blank (for trigram: candidate → word_after → word_after2)

        with open(self.input_filename, 'r', encoding='utf-8') as fin:
            text = fin.read()

        # clean the text of punctuation except for spaces and dashes
        text_cleaned = re.sub(r'[^\w\s-]', '', text)

        # Find all blank positions
        blank_pattern = re.compile(r'_{10,}')
        for match in blank_pattern.finditer(text_cleaned):
            blank_start = match.start()
            blank_end = match.end()

            # Get text before and after blank
            text_before = text_cleaned[:blank_start].strip()
            text_after = text_cleaned[blank_end:].strip()

            # Get words before blank
            words_before_text = text_before.split()
            if len(words_before_text) >= 1:
                word_before.append(words_before_text[-1].lower())
            else:
                word_before.append(None)
            if len(words_before_text) >= 2:
                word_before2.append(words_before_text[-2].lower())
            else:
                word_before2.append(None)

            # Get words after blank
            words_after_text = text_after.split()
            if len(words_after_text) >= 1:
                word_after.append(words_after_text[0].lower())
            else:
                word_after.append(None)
            if len(words_after_text) >= 2:
                word_after2.append(words_after_text[1].lower())
            else:
                word_after2.append(None)

        return word_before, word_after, word_before2, word_after2

    def _init_ngram_counts(self) -> None:
        word_before_set = set([w for w in self.word_before if w is not None])
        word_after_set = set([w for w in self.word_after if w is not None])
        word_before2_set = set([w for w in self.word_before2 if w is not None])
        word_after2_set = set([w for w in self.word_after2 if w is not None])
        candidates_words_set = set([w.lower() for w in self.candidates_words])

        with open(self.corpus_filename, 'r', encoding='utf-8') as fin:
            print('creating bigram and trigram from corpus ...')
            for i, line in enumerate(fin):
                words = self._tokenize(line)
                # Count unigrams
                self.unigrams.update(words)

                if len(words) < 2:
                    continue

                # Count bigrams once
                pair_counts = Counter(zip(words, words[1:]))

                # Loop only over bigrams that actually appear
                for (w1, w2), count in pair_counts.items():
                    # Pattern 1: word_before → candidate
                    if w1 in word_before_set and w2 in candidates_words_set:
                        self.bigrams[(w1, w2)] += count
                    # Pattern 2: candidate → word_after
                    elif not self.left_only and w1 in candidates_words_set and w2 in word_after_set:
                        self.bigrams[(w1, w2)] += count

                # Count trigrams once
                if len(words) >= 3:
                    triple_counts = Counter(zip(words, words[1:], words[2:]))

                    # Loop only over trigrams that actually appear
                    for (w1, w2, w3), count in triple_counts.items():
                        # Pattern 1: word_before2 → word_before → candidate
                        if w1 in word_before2_set and w2 in word_before_set and w3 in candidates_words_set:
                            self.trigrams[(w1, w2, w3)] += count
                        # Pattern 2: candidate → word_after → word_after2
                        elif not self.left_only and w1 in candidates_words_set and w2 in word_after_set and w3 in word_after2_set:
                            self.trigrams[(w1, w2, w3)] += count

                if i % 100000 == 0:
                    print(f"Finished {i} lines...")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, handling punctuation."""
        # Regular expression to remove punctuation
        punctuation_re = re.compile(r'[^\w\s-]')
        clean_text = punctuation_re.sub('', text).lower()
        words = clean_text.split()
        return words

    def _get_context(self, text: str, blank_pos: int) -> Tuple[List[str], List[str]]:
        """Extract left and right context around a blank position."""
        # Find the position of the blank in tokenized words
        # We need to count words before the blank
        text_before_blank = text[:blank_pos]
        words_before = self._tokenize(text_before_blank)

        # Get left context (last 2 words for trigram, last 1 for bigram)
        left_context = words_before[-2:] if len(words_before) >= 2 else words_before

        # Get right context if not left_only
        right_context = []
        if not self.left_only:
            text_after_blank = text[blank_pos:]
            # Remove the blank marker and get next words
            text_after_blank = text_after_blank.replace('__________', '', 1)
            words_after = self._tokenize(text_after_blank)
            right_context = words_after[:2] if len(words_after) >= 2 else words_after

        return left_context, right_context

    def _score_candidate_bigram(self, candidate: str, left_context: List[str], right_context: List[str]) -> float:
        """Score a candidate word using bigram model."""
        candidate_lower = candidate.lower()
        score = 0.0

        # Left bigram: (prev_word, candidate)
        if len(left_context) >= 1:
            prev_word = left_context[-1]
            bigram = (prev_word, candidate_lower)
            bigram_count = self.bigrams[bigram]
            unigram_count = self.unigrams[prev_word]
            if unigram_count > 0:
                score += bigram_count / unigram_count

        # Right bigram: (candidate, next_word) - only if not left_only
        if not self.left_only and len(right_context) >= 1:
            next_word = right_context[0]
            bigram = (candidate_lower, next_word)
            bigram_count = self.bigrams[bigram]
            unigram_count = self.unigrams[candidate_lower]
            if unigram_count > 0:
                score += bigram_count / unigram_count

        return score

    def _score_candidate_trigram(self, candidate: str, left_context: List[str], right_context: List[str]) -> float:
        """Score a candidate word using trigram model."""
        candidate_lower = candidate.lower()
        score = 0.0

        # Left trigram: (word1, word2, candidate)
        if len(left_context) >= 2:
            word1, word2 = left_context[-2], left_context[-1]
            trigram = (word1, word2, candidate_lower)
            trigram_count = self.trigrams[trigram]
            bigram_count = self.bigrams[(word1, word2)]
            if bigram_count > 0:
                score += trigram_count / bigram_count

        # Left bigram fallback: (word2, candidate)
        elif len(left_context) >= 1:
            word2 = left_context[-1]
            bigram = (word2, candidate_lower)
            bigram_count = self.bigrams[bigram]
            unigram_count = self.unigrams[word2]
            if unigram_count > 0:
                score += bigram_count / unigram_count

        # Right trigram: (candidate, next_word1, next_word2) - only if not left_only
        if not self.left_only and len(right_context) >= 2:
            next_word1, next_word2 = right_context[0], right_context[1]
            trigram = (candidate_lower, next_word1, next_word2)
            trigram_count = self.trigrams[trigram]
            bigram_count = self.bigrams[(candidate_lower, next_word1)]
            if bigram_count > 0:
                score += trigram_count / bigram_count

        # Right bigram fallback: (candidate, next_word1) - only if not left_only
        elif not self.left_only and len(right_context) >= 1:
            next_word1 = right_context[0]
            bigram = (candidate_lower, next_word1)
            bigram_count = self.bigrams[bigram]
            unigram_count = self.unigrams[candidate_lower]
            if unigram_count > 0:
                score += bigram_count / unigram_count

        return score

    def _score_candidate(self, candidate: str, left_context: List[str], right_context: List[str]) -> float:
        """Score a candidate using both bigram and trigram models."""
        trigram_score = self._score_candidate_trigram(candidate, left_context, right_context)
        bigram_score = self._score_candidate_bigram(candidate, left_context, right_context)

        # Combine scores (trigram gets higher weight as it's more specific)
        combined_score = 2.0 * trigram_score + bigram_score

        return combined_score

    def solve_cloze(self) -> List[str]:
        """Solve the cloze by finding the best candidate for each blank."""
        with open(self.input_filename, 'r', encoding='utf-8') as input_file:
            text = input_file.read()

        # Find all blank positions
        blank_pattern = re.compile(r'_{10,}')
        blanks = []
        for match in blank_pattern.finditer(text):
            blanks.append(match.start())

        solution = []

        for blank_pos in blanks:
            left_context, right_context = self._get_context(text, blank_pos)

            best_candidate = None
            best_score = float('-inf')

            # Score each candidate
            for candidate in self.candidates_words:
                score = self._score_candidate(candidate, left_context, right_context)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            # If no good match found, use first candidate as fallback
            if best_candidate is None:
                best_candidate = self.candidates_words[0] if self.candidates_words else ""

            solution.append(best_candidate)
            print(f'Blank at position {blank_pos}: selected "{best_candidate}" (score: {best_score:.6f})')

        return solution

    def solve_cloze_randomly(self) -> List[str]:
        return random.sample(self.candidates_words, len(self.candidates_words))

    def calculate_solution_accuracy(self, solution: List[str]):
        correct_order = self._get_candidates_words()
        matches = sum(1 for prediction, correct in zip(solution, correct_order) if prediction == correct)
        return (matches / len(correct_order)) * 100