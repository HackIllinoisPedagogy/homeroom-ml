import torch
import stanza

from scipy import spatial
from sentence_transformers import SentenceTransformer


class Tutor(object):
    """Automated tutor that guides student through problem."""

    def __init__(self, n_sub_hint, soln=None):
        super(Tutor, self).__init__()
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.nlp = stanza.Pipeline('en')
        self.n_sub_hint = n_sub_hint

        if soln is not None:
            self.load_soln(soln)

        self.curr_user_state = -1
        self.num_in_state = -1
        self.try_tree = True


    def load_soln(self, soln):  # Given a solution, load it into the tutor
        self.soln = soln
        self.soln_sep = soln.split('.')
        self.sentence_embeddings = self.model.encode(self.soln_sep)


    def get_similarity(self, a, b):
        return abs(1 - spatial.distance.cosine(a, b))


    def find_salient(self, sentence, sentence_embedding=None, tokens_sep=None):
        if tokens_sep is None:
            tokens_sep = sentence.split(" ")

        token_embeddings = self.model.encode(tokens_sep)

        if sentence_embedding is None:
            sentence_embedding = self.model.encode(sentence)

        sim_arr = []
        for token_embedding in token_embeddings:
            sim_arr.append(self.get_similarity(token_embedding, sentence_embedding))

        idxs = list(range(len(tokens_sep)))
        idxs.sort(key=lambda x: -sim_arr[x])

        return idxs


    def find_closest(self, curr_embedding):
        min_dist = None
        min_idx = 0

        for i in range(len(self.sentence_embeddings)):
            sent_embedding = self.sentence_embeddings[i]
            sim = self.get_similarity(curr_embedding, sent_embedding)

            if min_dist is None or sim > min_dist:
                min_dist = sim
                min_idx = i

        return min_idx


    def get_user_hint(self, user_input):
        curr_embedding = self.model.encode(user_input)
        user_state = self.find_closest(curr_embedding)

        if user_state == self.curr_user_state:
            self.num_in_state += 1
        else:
            self.curr_user_state = user_state
            self.num_in_state = 1
            self.try_tree = True

        if user_state == len(self.soln_sep)-1:
            return "You're very close!"

        hint_idx = user_state + 1
        sub_hint = self.get_sub_hint(hint_idx)

        return sub_hint


    def get_possible_keywords(self, doc):
        possible = []

        for sent in doc.sentences:
            for word in sent.words:
                if word.upos in set(["PROPN", "NOUN", "VERB"]):
                    possible.append(word)

        return possible


    def make_graph(self, doc):
        graph = {}

        for sent in doc.sentences:
            for word in sent.words:
                h = sent.words[word.head - 1] if word.head != 0 else None

                if h is not None:
                    if h not in graph:
                        graph[h] = []
                    graph[h].append(word)

        return graph


    def get_sub_hint(self, hint_idx):
        # First try to do the 'smarter' dependency-tree based hint generation
        doc = self.nlp(hint_idx)
        possible_keywords = self.get_possible_keywords(doc)

        if self.num_in_state > len(possible_keywords) and self.try_tree:
            self.try_tree = False
            self.num_in_state = 1

        if len(possible_keywords) > 0 and self.try_tree:
            graph = self.make_graph(doc)
            hint_node = possible_keywords[self.num_in_state - 1]




        # Find the most salient tokens of hint_idx
        tokens_sep = self.soln_sep[hint_idx].split(" ")
        salient_ids = self.find_salient(self.soln_sep[hint_idx],
            sentence_embedding=self.sentence_embeddings[hint_idx], tokens_sep=tokens_sep)

        # What is the length of the sub-hint we should give?
        len_hint = min(self.num_in_state, self.n_sub_hint) * len(salient_ids) / self.n_sub_hint
        to_include = [salient_ids[0]]
        s_id = 1

        while max(to_include) - min(to_include) + 1 < len_hint:
            to_include.append(salient_ids[s_id])
            s_id += 1

        return " ".join(tokens_sep[min(to_include): max(to_include) + 1])


    def reset_user_state(self):
        self.num_in_state = -1
        self.curr_user_state = -1


if __name__ == '__main__':
    soln = open("solution.txt").read()
    tutor = Tutor(3, soln=soln)

    user_state = "I know it's a geometric series, so the ratio should be log_2(x)/log_4(x). But I'm not sure what to do next"
    for i in range(5):
        print(tutor.get_user_hint(user_state))
