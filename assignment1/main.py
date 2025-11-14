import json
import time
from assignment1.cloze_solver import ClozeSolver

def solve_cloze(input, candidates, corpus, left_only):
    print(f'starting to solve the cloze {input} with {candidates} using {corpus}')
    solver = ClozeSolver(input_filename=input,
                         candidates_filename=candidates,
                         corpus_filename=corpus,
                         left_only=left_only,
                         max_ngram_order=5)
    solver.train()
    solution =  solver.solve_cloze()
    accuracy = solver.calculate_solution_accuracy(solution)
    print(f'cloze solved with accuracy: {accuracy:.2f}%')
    print(f'solving this cloze randomly would give an accuracy of: {solver.get_random_word_selection_accuracy():.2f}%')
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
