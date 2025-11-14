from typing import List, Tuple, Set, Dict
import re
import os
import pickle
from collections import Counter
import random


#TODO: add docstrings
#TODO: Move this class to main.py file as not allowed to submit multiple files
class ClozeSolver:
    def __init__(self,
                 input_filename: str,
                 candidates_filename: str,
                 corpus_filename: str,
                 left_only: bool,
                 max_ngram_order: int):

        self.input_filename = input_filename
        self.candidates_filename = candidates_filename
        self.corpus_filename = corpus_filename
        self.left_only = left_only
        self.max_ngram_order = max_ngram_order
        self.candidates_words = self._get_candidates_words()
        self.context_words = self._get_target_words()
        # Build n-gram models - dynamically create counters for each order
        self.ngrams = {n: Counter() for n in range(2, max_ngram_order + 1)}
        self.unigrams = Counter()

    def _get_candidates_words(self) -> List[str]:
        with open(self.candidates_filename, 'r', encoding='utf-8') as candidates_file:
            return candidates_file.read().splitlines()

    def _get_target_words(self) -> Dict[str, List[str]]:
        """
        Extract context words around each blank. Returns dict with lists for each position.
        
        Returns:
            dict: A dictionary with keys like 'before0', 'before1', 'after0', 'after1', etc.
                  Each key maps to a list where each element corresponds to a blank position.
                  - 'before0' contains the immediate word before each blank
                  - 'before1' contains the second word before each blank
                  - 'after0' contains the immediate word after each blank
                  - 'after1' contains the second word after each blank
                  - etc. for higher-order n-grams
                  - if there is no word available at a position (e.g., blank at start/end of text), None is appended to maintain consistent list lengths.
        """
        context = {}

        # For n-grams up to max_ngram_order, we need (max_ngram_order-1) words before/after
        for i in range(self.max_ngram_order - 1):
            context[f'before{i}'] = []
            context[f'after{i}'] = []

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
            for i in range(self.max_ngram_order - 1):
                idx = len(words_before_text) - 1 - i
                if idx >= 0:
                    context[f'before{i}'].append(words_before_text[idx].lower())
                else:
                    # No word available at this position (e.g., blank at start of sentence)
                    # These None values are filtered out during n-gram counting.
                    context[f'before{i}'].append(None)

            # Get words after blank
            words_after_text = text_after.split()
            for i in range(self.max_ngram_order - 1):
                if i < len(words_after_text):
                    context[f'after{i}'].append(words_after_text[i].lower())
                else:
                    # No word available at this position (e.g., blank at end of sentence)
                    # These None values are filtered out during n-gram counting.
                    context[f'after{i}'].append(None)

        return context

    def train(self):
        # TODO: delete _load_ngram_counts function and use only _init_ngram_counts as cant submit the pickle files
        self._load_ngram_counts()
        # self._init_ngram_counts()

    def solve_cloze(self) -> List[str]:
        """Solve the cloze by finding the best candidate for each blank."""
        with open(self.input_filename, 'r', encoding='utf-8') as input_file:
            text = input_file.read()

        blank_positions = self._find_blank_positions(text)
        solution = []

        for blank_pos in blank_positions:
            best_candidate, best_score = self._find_best_candidate_for_blank(text, blank_pos)
            solution.append(best_candidate)
            print(f'Blank at position {blank_pos}: selected "{best_candidate}" (score: {best_score:.6f})')

        return solution

    def _find_blank_positions(self, text: str) -> List[int]:
        """
        Find all blank positions in the text.

        Args:
            text: The input text containing blanks

        Returns:
            List of blank start positions
        """
        blank_pattern = re.compile(r'_{10,}')
        blanks = []
        for match in blank_pattern.finditer(text):
            blanks.append(match.start())
        return blanks

    def _find_best_candidate_for_blank(self, text: str, blank_pos: int) -> Tuple[str, float]:
        """
        Find the best candidate word for a single blank position.

        Args:
            text: The input text containing blanks
            blank_pos: The start position of the blank

        Returns:
            Tuple of (best_candidate_word, best_score)
        """
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

        return best_candidate, best_score

    def _get_context(self, text: str, blank_pos: int) -> Tuple[List[str], List[str]]:
        """Extract left and right context around a blank position."""
        text_before_blank = text[:blank_pos]
        words_before = self._tokenize(text_before_blank)

        # Get left context (up to max_ngram_order-1 words)
        left_context = words_before[-(self.max_ngram_order - 1):] if len(words_before) >= (
                    self.max_ngram_order - 1) else words_before

        # Get right context if not left_only
        right_context = []
        if not self.left_only:
            text_after_blank = text[blank_pos:]
            text_after_blank = text_after_blank.replace('__________', '', 1)
            words_after = self._tokenize(text_after_blank)
            right_context = words_after[:self.max_ngram_order - 1] if len(words_after) >= (
                        self.max_ngram_order - 1) else words_after

        return left_context, right_context

    def _score_candidate(self, candidate: str, left_context: List[str], right_context: List[str]) -> float:
        """Score a candidate using all n-gram models, with higher weights for higher-order n-grams."""
        combined_score = 0.0

        # Score with each n-gram order, with increasing weights
        for n in range(2, self.max_ngram_order + 1):
            ngram_score = self._score_candidate_ngram(candidate, left_context, right_context, n)
            # Higher-order n-grams get exponentially higher weights
            weight = (n - 1) ** 2  # 1, 4, 9, 16 for bigrams, trigrams, 4-grams, 5-grams
            combined_score += weight * ngram_score

        return combined_score

    def _score_candidate_ngram(self,
                               candidate: str,
                               left_context: List[str],
                               right_context: List[str],
                               n: int) -> float:
        """Score a candidate word using n-gram model of order n."""
        candidate_lower = candidate.lower()
        score = 0.0

        # Left n-gram: (context_words..., candidate)
        if len(left_context) >= n - 1:
            ngram_tuple = tuple(left_context[-(n - 1):] + [candidate_lower])
            ngram_count = self.ngrams[n][ngram_tuple]

            # Get (n-1)-gram count for normalization
            if n > 2:
                prev_ngram = tuple(left_context[-(n - 1):])
                prev_count = self.ngrams[n - 1][prev_ngram]
            else:
                # For bigrams, use unigram count
                prev_count = self.unigrams[left_context[-1]] if left_context else 0

            if prev_count > 0:
                score += ngram_count / prev_count
        # Fallback to lower-order n-gram
        elif len(left_context) >= 1 and n > 2:
            return self._score_candidate_ngram(candidate, left_context, right_context, n - 1)

        # Right n-gram: (candidate, context_words...) - only if not left_only
        if not self.left_only and len(right_context) >= n - 1:
            ngram_tuple = tuple([candidate_lower] + right_context[:n - 1])
            ngram_count = self.ngrams[n][ngram_tuple]

            # Get (n-1)-gram count for normalization
            if n > 2:
                prev_ngram = tuple([candidate_lower] + right_context[:n - 2])
                prev_count = self.ngrams[n - 1][prev_ngram]
            else:
                # For bigrams, use unigram count
                prev_count = self.unigrams[candidate_lower]

            if prev_count > 0:
                score += ngram_count / prev_count
        # Fallback to lower-order n-gram
        elif not self.left_only and len(right_context) >= 1 and n > 2:
            score += self._score_candidate_ngram(candidate, left_context, right_context, n - 1)

        return score

    def _load_ngram_counts(self):
        pickle_files = {n: f'{n}grams.pkl' for n in range(2, self.max_ngram_order + 1)}
        pickle_files['unigrams'] = 'unigrams.pkl'

        all_exist = all(os.path.isfile(f) for f in pickle_files.values())

        if not all_exist:
            self._init_ngram_counts()
            print("\nsaving unigrams to file ...")
            pickle.dump(self.unigrams, open('unigrams.pkl', 'wb'))
            print("finished saving unigrams to file ...")
            for n in range(2, self.max_ngram_order + 1):
                print(f"\nsaving {n}grams to file ...")
                pickle.dump(self.ngrams[n], open(f'{n}grams.pkl', 'wb'))
                print(f"finished saving {n}grams to file ...")
        else:
            print("\nloading unigrams pkl ...")
            self.unigrams = pickle.load(open('unigrams.pkl', 'rb'))
            print("loaded unigrams pkl ...")
            for n in range(2, self.max_ngram_order + 1):
                print(f"\nloading {n}grams pkl ...")
                self.ngrams[n] = pickle.load(open(f'{n}grams.pkl', 'rb'))
                print(f"loaded {n}grams pkl ...")

    # TODO: make the _init_ngram_counts function async to speed up the training process
    def _init_ngram_counts(self) -> None:
        """Initialize n-gram counts from corpus, only counting relevant n-grams."""
        # Build sets for fast lookup
        self.context_sets: Dict[str, Set[str]] = {}
        for key, word_list in self.context_words.items():
            # Filter out None values which indicate missing context words
            self.context_sets[key] = set([w for w in word_list if w is not None])
        self.candidates_words_set = set([w.lower() for w in self.candidates_words])

        with open(self.corpus_filename, 'r', encoding='utf-8') as fin:
            print(f'creating n-grams (up to {self.max_ngram_order}) from corpus ...')
            for i, line in enumerate(fin):
                words = self._tokenize(line)
                # Count unigrams
                self.unigrams.update(words)

                # Count n-grams for each order
                for n in range(2, self.max_ngram_order + 1):
                    self._update_ngrams(words, n)

                if i % 100000 == 0:
                    print(f"Finished {i} lines...")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, handling punctuation."""
        # Regular expression to remove punctuation
        punctuation_re = re.compile(r'[^\w\s-]')
        clean_text = punctuation_re.sub('', text).lower()
        return clean_text.split()

    def _update_ngrams(self, words: List[str], n: int) -> None:
        """
        Process and count n-grams of order n for a line of text.
        Only counts n-grams that match the relevant patterns (before→candidate or candidate→after).

        Args:
            words: Tokenized words from a line
            n: The order of n-grams to process
        """
        if len(words) < n:
            return

        # Count n-grams once using Counter
        ngram_counts = Counter(zip(*[words[j:] for j in range(n)]))

        # Loop only over n-grams that actually appear
        for ngram_tuple, count in ngram_counts.items():
            # Pattern 1: context_before → ... → candidate
            if self._matches_before_pattern(ngram_tuple, n):
                self.ngrams[n][ngram_tuple] += count

            # Pattern 2: candidate → context_after → ... (only if not left_only)
            elif not self.left_only and self._matches_after_pattern(ngram_tuple, n):
                self.ngrams[n][ngram_tuple] += count

    def _matches_before_pattern(self, ngram_tuple: tuple, n: int) -> bool:
        """
        Check if an n-gram matches pattern 1: context_before → ... → candidate.

        Args:
            ngram_tuple: The n-gram tuple to check
            n: The order of the n-gram

        Returns:
            bool: True if the n-gram matches the before pattern
        """
        # Check if first (n-1) words match context_before patterns and last word is candidate
        for j in range(n - 1):
            context_key = f'before{j}'
            if ngram_tuple[j] not in self.context_sets.get(context_key, set()):
                return False
        return ngram_tuple[-1] in self.candidates_words_set

    def _matches_after_pattern(self, ngram_tuple: tuple, n: int) -> bool:
        """
        Check if an n-gram matches pattern 2: candidate → context_after → ...

        Args:
            ngram_tuple: The n-gram tuple to check
            n: The order of the n-gram

        Returns:
            bool: True if the n-gram matches the after pattern
        """
        # Check if first word is candidate and remaining words match context_after patterns
        if ngram_tuple[0] not in self.candidates_words_set:
            return False

        for j in range(1, n):
            context_key = f'after{j - 1}'
            if ngram_tuple[j] not in self.context_sets.get(context_key, set()):
                return False
        return True

    def get_random_word_selection_accuracy(self, num_of_random_solutions: int = 1000) -> float:
        """
        Generates random cloze solutions and returns the mean accuracy.

        Args:
            num_of_random_solutions (int, optional): Number of random solutions to generate.
                Defaults to 100.

        Returns:
            float: Mean accuracy (percentage) of random solutions.
        """
        accuracies = []
        for _ in range(num_of_random_solutions):
            random_solution = self.solve_cloze_randomly()
            accuracy = self.calculate_solution_accuracy(random_solution)
            accuracies.append(accuracy)

        return sum(accuracies) / len(accuracies)

    def solve_cloze_randomly(self) -> List[str]:
        return random.choices(self.candidates_words, k=len(self.candidates_words))

    def calculate_solution_accuracy(self, solution: List[str]):
        correct_order = self.candidates_words # The correct order is the order in candidates file
        matches = sum(1 for prediction, correct in zip(solution, correct_order) if prediction == correct)
        return (matches / len(correct_order)) * 100