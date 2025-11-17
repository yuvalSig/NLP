# NLP
Natural Processing Language Course 

# 🧩 Task 1 — Heaps’ Law & Zipf’s Law

### ✔️ Zipf’s Law
The file **plot_heaps_zipf_laws.py** includes the original implementation for plotting Zipf’s distribution.  
Running it produces a log-log graph of word frequency vs. rank for the Wikipedia 2018 dataset fragment.

### ✔️ Heaps’ Law (implemented by me)
I added a new function.

The result: 

<img width="865" height="649" alt="image" src="https://github.com/user-attachments/assets/9ac55d47-6cb2-4409-ac9c-7983b07477eb" />
<img width="865" height="649" alt="image" src="https://github.com/user-attachments/assets/f7ab3819-18b6-40de-859d-2167aede866d" />




# ▶️ How to Run

### 1. **Download the corpus**
Download the 10M-line 2018 Wikipedia dataset from the assignment instructions:  
https://drive.google.com/file/d/15H-pg40Epx2u6GE14vquCdF_-VlOgOG7/view?usp=sharing

Place the file inside:

### 2. **Configure paths**
Edit `config.json` if needed:

```json
{
  "corpus": "assignment1/data/en.wikipedia2018.10M.txt",
  "input_filename": "assignment1/data/document.cloze.txt",
  "candidates_filename": "assignment1/data/candidate.words.txt",
  "left_only": false
}

python assignment1/plot_heaps_zipf_laws.py

