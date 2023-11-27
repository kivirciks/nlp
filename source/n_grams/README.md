## Лабораторная работа №2. N-grams
**Задачи:**
<li> Извлечь триграммы из текста на основании своего алгоритма и на основании скрипта, основанного на библиотеке `nltk` <br>
<li> Реализовать алгоритм расчета меры ассоциации лексемы (`MI` и `log-likelihood`)<br>

**Меры ассоциативной связанности (`association measures`)** - меры, вычисляющие силу связи между элементами в составе коллокации (параметры: частота совместной встречаемости, частота слова в корпусе, размер корпуса, и др.).  

**MI**
![MI](https://github.com/kivirciks/nlp/blob/main/assets/MI.png)
`n` — ключевое слово (`node`);  
`c` — коллокат (`collocate`);  
`f (n, c)` — частота встречаемости ключевого слова `n` в паре с коллокатом `c`;  
`f(n)`, `f(c)` — абсолютные (независимые) частоты ключевого слова `n` и слов `c` в корпусе (тексте) соответственно;  
`N` — общее число словоупотреблений в корпусе (тексте);  
`ngram` - количество слов в `n`-грамме (например, для триграмм `ngram = 3`);  
`f(u_i)` - абсолютная частота `i`-й униграммы в `n`-грамме.  

**Log-likelihood**
![log-likelihood](https://github.com/kivirciks/nlp/blob/main/assets/log-likelihood.png)
`O_ij`, `E_ij` - наблюдаемая и ожидаемая частоты соответственно;  
`ngram` - количество слов в `n`-грамме (например, для триграмм `ngram = 3`).  

**N-grams без использования `nltk`**
Пример кода для извлечение триграмм из текста на основании лексем

```python
def extract_trigrams(text):
    trigrams = []
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    for i in range(len(lemmas) - 2):
        trigram = ' '.join([lemmas[i], lemmas[i+1], lemmas[i+2]])
        trigrams.append(trigram)
    return trigrams
```

Пример кода для подсчета MI

```python
def calculate_mutual_information(trigram_counts, trigram, unigram_counts):
    word1, word2, word3 = trigram.split()
    unigram1_count = unigram_counts[word1]
    unigram2_count = unigram_counts[word2]
    unigram3_count = unigram_counts[word3]
    trigram_count = trigram_counts[trigram]
    total_count = sum(trigram_counts.values())
    mutual_information = math.log2((trigram_count*total_count) / (unigram1_count*unigram2_count*unigram3_count))
    return mutual_information
```
Пример кода для подсчета log likelihood

```python
def calculate_log_likelihood(trigram_counts, trigram, unigram_counts):
    word1, word2, word3 = trigram.split()
    unigram1_count = unigram_counts[word1]
    unigram2_count = unigram_counts[word2]
    unigram3_count = unigram_counts[word3]
    trigram_count = trigram_counts[trigram]
    total_count = sum(trigram_counts.values())
    expected_count = (unigram1_count*unigram2_count*unigram3_count) / total_count
    log_likelihood = 2 * (trigram_count * math.log2(trigram_count / expected_count))
    return log_likelihood
```



**N-grams с использованием `nltk`**
Пример кода для извлечение триграмм из текста и подсчет метрик на основании лексем

```python
trigram_measures = nltk.collocations.TrigramAssocMeasures()
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
finder_thr = TrigramCollocationFinder.from_words(text)
trigram_scores = finder_thr.score_ngrams(trigram_measures.pmi)
```

ТОП-30 триграмм 

| Триграмма | MI | Column 3 |
|----------|----------|----------|
| abend einem offiziellen    | 5.4297629888288625   | Cell 3   |
| bainum outspent mufi    | 5.4297629888288625   | Cell 6   |
| circusif roone arledge    | 5.4297629888288625   | Cell 9   |
| claptons laylarecently guitarworld    | 5.4297629888288625   | Cell 9   |
| desa rueng bakjok    | 5.4297629888288625   | Cell 9   |
| dropper downloader bagleaq    | 5.4297629888288625   | Cell 9   |
| easybus easypizza easycar    | 5.4297629888288625   | Cell 9   |
| electionbefore shugofa beheshti    | 5.4297629888288625   | Cell 9   |
| fadhil muhsen salom    | 5.4297629888288625   | Cell 9   |
| gimm mur schwg    | 5.4297629888288625   | Cell 9   |
| hanno rilasciato congiuntamente    | 5.4297629888288625   | Cell 9   |
| jasbir kang yuba    | 5.4297629888288625   | Cell 9   |
| lifesized jabba hutt    | 5.4297629888288625   | Cell 9   |
| madame edmey cimeus    | 5.4297629888288625   | Cell 9   |
| namesas bassist popfunk    | 5.4297629888288625   | Cell 9   |
| newburg bahar uttam    | 5.4297629888288625   | Cell 9   |
| olympicsby miron varouhakis    | 5.4297629888288625   | Cell 9   |
| parmi lesquelles figurent    | 5.4297629888288625   | Cell 9   |
| quebeckers guylaine dumont    | 5.4297629888288625   | Cell 9   |
| raad altamimi newlylaunched    | 5.4297629888288625   | Cell 9   |
| staphylococcus aureus mrsa    | 5.4297629888288625   | Cell 9   |
| technogadgets holidaystalk overachiever    | 5.4297629888288625   | Cell 9   |
| undoing flintoffbrian laras    | 5.4297629888288625   | Cell 9   |
| utils bsd licenseand    | 5.4297629888288625   | Cell 9   |
| waisted emeraldeyed brunette    | 5.4297629888288625   | Cell 9   |
| xwiki aspwiki snipsnap    | 5.4297629888288625   | Cell 9   |
| yearjune kamila vodichkova    | 5.4297629888288625   | Cell 9   |
| zhah stoyah kohvich    | 5.4297629888288625   | Cell 9   |
| zihdroo nuhs ihlgows    | 5.4297629888288625   | Cell 9   |
| zservices clientsofts servicebuilder    | 5.4297629888288625   | Cell 9   |
