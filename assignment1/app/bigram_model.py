import random
from collections import defaultdict, Counter

class BigramModel:
    def __init__(self, corpus):
        """
        Initialize the model with a list of sentences (strings).
        """
        self.corpus = corpus
        self.bigrams = defaultdict(Counter)
        self.vocab = set()
        self._build()

    def _tokenize(self, sentence):
        # Very simple tokenizer (split on spaces, lowercase)
        tokens = sentence.lower().split()
        return ["<s>"] + tokens + ["</s>"]

    def _build(self):
        for sentence in self.corpus:
            tokens = self._tokenize(sentence)
            self.vocab.update(tokens)
            for w1, w2 in zip(tokens[:-1], tokens[1:]):
                self.bigrams[w1][w2] += 1
        # Convert counts to probabilities
        self.bigram_probs = {
            w1: {w2: count / sum(counter.values())
                 for w2, count in counter.items()}
            for w1, counter in self.bigrams.items()
        }

    def next_word_distribution(self, word):
        """
        Return probability distribution of next words after given word.
        """
        return self.bigram_probs.get(word.lower(), {})

    def generate_word(self, word):
        """
        Sample a single next word given the current word.
        """
        dist = self.next_word_distribution(word)
        if not dist:  # unseen word, backoff to <s>
            dist = self.next_word_distribution("<s>")
        words, probs = zip(*dist.items())
        return random.choices(words, probs)[0]

    def generate_text(self, start_word="<s>", length=10):
        """
        Generate a sequence of words of given length.
        """
        current = start_word
        result = []
        for _ in range(length):
            nxt = self.generate_word(current)
            if nxt == "</s>":
                break
            result.append(nxt)
            current = nxt
        return " ".join(result)

    def get_embedding(self, word, as_sparse=True, top_k=None):
        """
        Represent the word as its conditional next-word distribution.
        """
        dist = self.next_word_distribution(word)
        if not dist:  # unseen → backoff
            dist = self.next_word_distribution("<s>")

        if as_sparse:
            # Optionally keep top_k
            if top_k:
                dist = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_k])
            return dist
        else:
            # Dense vector over whole vocab
            return {w: dist.get(w, 0.0) for w in sorted(self.vocab)}