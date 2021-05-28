from pagerank import sample_pagerank, crawl

corpus = crawl("corpus2")
damp = 0.85
result = sample_pagerank(corpus, damp, 30)
print(result)