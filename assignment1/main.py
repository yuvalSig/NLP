import json
import time
from typing import List, Tuple, Set, Dict
import re
import math
from collections import Counter
import random
from multiprocessing import Pool, cpu_count


class ClozeSolver:
    """
    Solves cloze tests using n-gram language models.

    Trains n-gram models on a corpus by counting n-grams that match relevant patterns.
    Uses Laplace smoothing and log probabilities for scoring candidates.

    Main methods:
    - train(): Train n-gram models from corpus
    - solve_cloze(): Find best candidate for each blank
    - calculate_solution_accuracy(): Evaluate solution accuracy
    """

    def __init__(self,
                 input_filename: str,
                 candidates_filename: str,
                 corpus_filename: str,
                 left_only: bool,
                 max_ngram_order: int):
        """
        Initialize the ClozeSolver with input files and configuration.

        Args:
            input_filename: Path to the cloze document file containing blanks
            candidates_filename: Path to the file containing candidate words
            corpus_filename: Path to the corpus file for training n-gram models
            left_only: If True, only use left context for scoring; if False, use both left and right context
            max_ngram_order: Maximum order of n-grams to use.
                Used it as a parameter for experimenting with different n-gram sizes and to see their effect on the accuracy.
        """
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
        self.vocabulary_size = 0  # Will be set after training

    def _get_candidates_words(self) -> List[str]:
        """
        Read candidate words from the candidates file.

        Returns:
            List of candidate words, one per line from the file
        """
        with open(self.candidates_filename, 'r', encoding='utf-8') as candidates_file:
            return candidates_file.read().splitlines()

    def _get_target_words(self) -> Dict[str, List[str]]:
        """
        Extract context words around each blank from the input file.

        Returns:
            Dict[str, List[str]]: A dictionary with keys like 'before0', 'before1', 'after0', 'after1', etc.
                                  Each key maps to a list where each element corresponds to a blank position.
                                  - 'before0' contains the immediate word before each blank
                                  - 'before1' contains the second word before each blank
                                  - 'after0' contains the immediate word after each blank
                                  - 'after1' contains the second word after each blank
                                  - etc. for higher-order n-grams
                                  - if there is no word available at a position (e.g., blank at start/end of text),
                                    None is appended to maintain consistent list lengths.
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
        """
        Train the n-gram models by initializing n-gram counts from the corpus.
        """
        self._init_ngram_counts()

    def solve_cloze(self, smoothing_k: float) -> List[str]:
        """
        Solve the cloze by finding the best candidate for each blank.

        Args:
            smoothing_k: Laplace smoothing parameter for scoring (add-k smoothing).

        Returns:
            List of candidate words for each blank position.
        """
        with open(self.input_filename, 'r', encoding='utf-8') as input_file:
            text = input_file.read()

        blank_positions = self._find_blank_positions(text)
        solution = []

        for blank_pos in blank_positions:
            best_candidate, best_score = self._find_best_candidate_for_blank(text, blank_pos, smoothing_k)
            solution.append(best_candidate)

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

    def _find_best_candidate_for_blank(self, text: str, blank_pos: int, smoothing_k: float) -> Tuple[str, float]:
        """
        Find the best candidate word for a single blank position.

        Args:
            text: The input text containing blanks
            blank_pos: The start position of the blank
            smoothing_k: Laplace smoothing parameter

        Returns:
            Tuple of (best_candidate_word, best_score)
        """
        left_context, right_context = self._get_context(text, blank_pos)

        best_candidate = None
        best_score = float('-inf')

        # Score each candidate
        for candidate in self.candidates_words:
            score = self._score_candidate(candidate, left_context, right_context, smoothing_k)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        # If no good match found, use first candidate as fallback
        if best_candidate is None:
            best_candidate = self.candidates_words[0] if self.candidates_words else ""

        return best_candidate, best_score

    def _get_context(self, text: str, blank_pos: int) -> Tuple[List[str], List[str]]:
        """
        Extract left and right context around a blank position.

        Args:
            text: The input text containing blanks
            blank_pos: The start position of the blank in the text

        Returns:
            Tuple[List[str], List[str]]: A tuple containing (left_context, right_context)
                                         where each is a list of words around the blank position
        """
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

    def _score_candidate(self,
                         candidate: str,
                         left_context: List[str],
                         right_context: List[str],
                         smoothing_k: float) -> float:
        """
        Score a candidate word using all n-gram models with smoothing (using log probabilities).

        Combines scores from all n-gram orders (bigrams through max_ngram_order) with exponentially
        increasing weights, where higher-order n-grams receive more weight.

        Args:
            candidate: The candidate word to score
            left_context: List of words before the blank position
            right_context: List of words after the blank position
            smoothing_k: Laplace smoothing parameter for probability estimation

        Returns:
            float: Combined log score for the candidate (higher is better)
        """
        combined_log_score = float('-inf')  # Start with log(0) = -inf

        # Score with each n-gram order, with increasing weights
        for n in range(2, self.max_ngram_order + 1):
            ngram_log_score = self._score_candidate_ngram(candidate, left_context, right_context, n, smoothing_k)
            if ngram_log_score == float('-inf'):  # No valid n-gram found, skip
                continue
            # Higher-order n-grams get exponentially higher weights
            weight = (n - 1) ** 2
            weighted_log_score = math.log(weight) + ngram_log_score
            combined_log_score = self._add_log_probabilities(combined_log_score, weighted_log_score)

        return combined_log_score

    def _score_candidate_ngram(self,
                               candidate: str,
                               left_context: List[str],
                               right_context: List[str],
                               n: int,
                               smoothing_k: float) -> float:
        """
        Score a candidate word using n-gram model of order n with Laplace smoothing (log probabilities).

        Scores the candidate by checking both left and right n-gram contexts (if right context is enabled).
        Falls back to lower-order n-grams if sufficient context is not available.

        Args:
            candidate: The candidate word to score
            left_context: List of words before the blank position
            right_context: List of words after the blank position
            n: The order of the n-gram model to use
            smoothing_k: Laplace smoothing parameter for probability estimation

        Returns:
            float: Log probability score for the candidate using n-grams of order n
        """
        candidate_lower = candidate.lower()
        log_score = float('-inf')  # Start with log(0) = -inf

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

            log_prob = self._calculate_smoothed_probability(ngram_count, prev_count, smoothing_k)
            log_score = self._add_log_probabilities(log_score, log_prob)
        # Fallback to lower-order n-gram
        elif len(left_context) >= 1 and n > 2:
            return self._score_candidate_ngram(candidate, left_context, right_context, n - 1, smoothing_k)

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

            log_prob = self._calculate_smoothed_probability(ngram_count, prev_count, smoothing_k)
            log_score = self._add_log_probabilities(log_score, log_prob)
        # Fallback to lower-order n-gram
        elif not self.left_only and len(right_context) >= 1 and n > 2:
            lower_log_score = self._score_candidate_ngram(candidate, left_context, right_context, n - 1, smoothing_k)
            log_score = self._add_log_probabilities(log_score, lower_log_score)

        return log_score

    def _add_log_probabilities(self, log_score: float, log_prob: float) -> float:
        """
        Add two log probabilities: log(exp(log_score) + exp(log_prob)).

        Args:
            log_score: Current log score (can be -inf)
            log_prob: New log probability to add (can be -inf)

        Returns:
            log(exp(log_score) + exp(log_prob))
        """
        # If either is -inf, return the other (or -inf if both are -inf)
        if log_score == float('-inf'):
            return log_prob
        if log_prob == float('-inf'):
            return log_score

        # Use log-sum-exp trick: log(a + b) = max(log_a, log_b) + log(1 + exp(min - max))
        max_log = max(log_score, log_prob)
        return max_log + math.log(1 + math.exp(min(log_score, log_prob) - max_log))

    def _calculate_smoothed_probability(self, ngram_count: int, prev_count: int, smoothing_k: float) -> float:
        """
        Calculate smoothed log probability using Laplace smoothing.

        Args:
            ngram_count: Count of the n-gram
            prev_count: Count of the (n-1)-gram context
            smoothing_k: Laplace smoothing parameter

        Returns:
            Log of smoothed probability: log((count + k) / (prev_count + k * V))
        """
        smoothed_prob = (ngram_count + smoothing_k) / (prev_count + smoothing_k * self.vocabulary_size)
        # Return log probability for numerical stability
        return math.log(smoothed_prob) if smoothed_prob > 0 else float('-inf')

    def _init_ngram_counts(self) -> None:
        """
        Initialize n-gram counts from corpus using multiprocessing for parallel processing.

        Reads the corpus file, splits it into chunks, and processes each chunk in parallel
        to count n-grams. Only counts n-grams that match the relevant patterns (context_before→candidate
        or candidate→context_after). Merges results from all chunks and sets the vocabulary_size attribute.
        """
        # Build sets for fast lookup
        context_sets = {}
        for key, word_list in self.context_words.items():
            # Filter out None values which indicate missing context words
            context_sets[key] = set([w for w in word_list if w is not None])
        candidates_words_set = set([w.lower() for w in self.candidates_words])

        print(f'creating n-grams (up to {self.max_ngram_order}) from corpus using multiprocessing...')

        # Read all lines
        with open(self.corpus_filename, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        # Determine number of workers
        num_workers = cpu_count() or 1  # Fallback to 1 if cpu_count() returns None, thus init the ngrams sequentially
        chunk_size = max(100000, len(lines) // num_workers)
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

        print(f"Processing {len(lines)} lines in {len(chunks)} chunks using {num_workers} workers...")
        processor = NgramProcessor(
            context_sets,
            candidates_words_set,
            self.left_only,
            self.max_ngram_order
        )
        # Process chunks in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.map(processor.process_chunk, enumerate(chunks))

        # Merge results from all chunks
        print("Merging results from all chunks...")
        for unigrams_chunk, ngrams_chunk in results:
            self.unigrams.update(unigrams_chunk)
            for n in range(2, self.max_ngram_order + 1):
                self.ngrams[n].update(ngrams_chunk[n])

        self.vocabulary_size = max(len(self.unigrams), 1)  # Ensure at least 1

        print(f"Finished processing {len(lines)} lines")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, handling punctuation."""
        # Regular expression to remove punctuation
        punctuation_re = re.compile(r'[^\w\s-]')
        clean_text = punctuation_re.sub('', text).lower()
        return clean_text.split()

    def get_random_word_selection_accuracy(self, num_of_random_solutions: int) -> float:
        """
        Generates random cloze solutions and returns the mean accuracy.

        Args:
            num_of_random_solutions (int): Number of random solutions to generate.

        Returns:
            float: Mean accuracy (percentage) of random solutions.
        """
        accuracies = []
        for _ in range(num_of_random_solutions):
            random_solution = self.solve_cloze_randomly()
            accuracy = self.calculate_solution_accuracy(random_solution)
            accuracies.append(accuracy)

        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    def solve_cloze_randomly(self) -> List[str]:
        """
        Generate a random solution by randomly selecting candidates for each blank.

        Returns:
            List[str]: List of randomly selected candidate words, one for each blank position
        """
        return random.choices(self.candidates_words, k=len(self.candidates_words))

    def calculate_solution_accuracy(self, solution: List[str]) -> float:
        """
        Calculate the accuracy of a solution by comparing it to the correct order.

        Args:
            solution: List of predicted candidate words for each blank position

        Returns:
            float: Accuracy percentage (0-100) of the solution
        """
        correct_order = self.candidates_words  # The correct order is the order in candidates file
        matches = sum(1 for prediction, correct in zip(solution, correct_order) if prediction == correct)
        return (matches / len(correct_order)) * 100


class NgramProcessor:
    """Helper class for multiprocessing that replicates the helper functions."""

    def __init__(self,
                 context_sets: Dict[str, Set[str]],
                 candidates_words_set: Set[str],
                 left_only: bool,
                 max_ngram_order: int):
        """
        Initialize the NgramProcessor for parallel processing of corpus chunks.

        Args:
            context_sets: Dictionary mapping context keys (e.g., 'before0', 'after1') to sets of context words
            candidates_words_set: Set of candidate words (lowercase)
            left_only: If True, only count n-grams matching the before→candidate pattern
            max_ngram_order: Maximum order of n-grams to process
        """
        self.context_sets = context_sets
        self.candidates_words_set = candidates_words_set
        self.left_only = left_only
        self.max_ngram_order = max_ngram_order
        self.punctuation_re = re.compile(r'[^\w\s-]')

    def process_chunk(self, chunk_data: Tuple[int, List[str]]) -> Tuple[Counter, Dict[int, Counter]]:
        """
        Process a chunk of lines from the corpus to count unigrams and n-grams.

        Args:
            chunk_data: Tuple of (chunk_index, list_of_lines) where each line is a string from the corpus

        Returns:
            Tuple[Counter, Dict[int, Counter]]: A tuple containing:
                - Counter of unigram counts
                - Dictionary mapping n-gram order to Counter of n-gram counts
        """
        chunk_idx, chunk = chunk_data
        unigrams_chunk = Counter()
        ngrams_chunk = {n: Counter() for n in range(2, self.max_ngram_order + 1)}

        for i, line in enumerate(chunk):
            words = self._tokenize(line)
            # Count unigrams
            unigrams_chunk.update(words)
            # Count n-grams for each order
            for n in range(2, self.max_ngram_order + 1):
                self._update_ngrams(words, n, ngrams_chunk)

            if (chunk_idx * len(chunk) + i) % 100000 == 0:
                print(f"Finished {chunk_idx * len(chunk) + i} lines...")

        return unigrams_chunk, ngrams_chunk

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words, handling punctuation.

        Args:
            text: The input text to tokenize

        Returns:
            List[str]: List of lowercase words with punctuation removed
        """
        clean_text = self.punctuation_re.sub('', text).lower()
        return clean_text.split()

    def _update_ngrams(self, words: List[str], n: int, ngrams_chunk: Dict[int, Counter]) -> None:
        """
        Process and count n-grams of order n for a line of text.
        Only counts n-grams that match the relevant patterns (before→candidate or candidate→after).

        Args:
            words: Tokenized words from a line
            n: The order of n-grams to process
            ngrams_chunk: Dictionary mapping n to Counter for storing n-gram counts
        """
        if len(words) < n:
            return
        # Count n-grams once using Counter
        ngram_counts = Counter(zip(*[words[j:] for j in range(n)]))

        # Loop only over n-grams that actually appear
        for ngram_tuple, count in ngram_counts.items():
            # Pattern 1: context_before → ... → candidate
            if self._matches_before_pattern(ngram_tuple, n):
                ngrams_chunk[n][ngram_tuple] += count
            # Pattern 2: candidate → context_after → ... (only if not left_only)
            elif not self.left_only and self._matches_after_pattern(ngram_tuple, n):
                ngrams_chunk[n][ngram_tuple] += count

    def _matches_before_pattern(self, ngram_tuple: tuple, n: int) -> bool:
        """
        Check if an n-gram matches pattern 1: context_before → ... → candidate.

        Args:
            ngram_tuple: The n-gram tuple to check
            n: The order of the n-gram

        Returns:
            bool: True if the n-gram matches the pattern where the first (n-1) words match
                  context_before patterns and the last word is a candidate word
        """
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
            bool: True if the n-gram matches the pattern where the first word is a candidate
                  and the remaining words match context_after patterns
        """
        if ngram_tuple[0] not in self.candidates_words_set:
            return False
        for j in range(1, n):
            context_key = f'after{j - 1}'
            if ngram_tuple[j] not in self.context_sets.get(context_key, set()):
                return False
        return True

def solve_cloze(input, candidates, corpus, left_only):
    print(f'starting to solve the cloze {input} with {candidates} using {corpus}')
    solver = ClozeSolver(input_filename=input,
                         candidates_filename=candidates,
                         corpus_filename=corpus,
                         left_only=left_only,
                         max_ngram_order=5)

    num_of_random_solutions = 1000
    print(f'solving this cloze randomly over {num_of_random_solutions} solutions would give an accuracy of: '
          f'{solver.get_random_word_selection_accuracy(num_of_random_solutions):.2f}%')

    solver.train()
    solution =  solver.solve_cloze(smoothing_k=0.00001)
    accuracy = solver.calculate_solution_accuracy(solution)
    print(f'cloze solved with accuracy: {accuracy:.2f}%')
    return solution


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['corpus'],
                           config['left_only'])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time:.2f} seconds")

    print('cloze solution:', solution)