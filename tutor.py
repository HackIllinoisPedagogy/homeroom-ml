import torch
from scipy import spatial
from sentence_transformers import SentenceTransformer

class Tutor(object):
    """Automated tutor that guides student through problem."""

    def __init__(self, soln=None):
        super(Tutor, self).__init__()
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

        if soln is not None:
            self.load_soln(soln)

    def load_soln(self, soln):  # Given a solution, load it into the tutor
        self.soln = soln
        self.soln_sep = soln.split('.')
        self.sentence_embeddings = self.model.encode(self.soln_sep)

    def get_similarity(self, a, b):
        return abs(1 - spatial.distance.cosine(a, b))

    def get_user_state(self, user_state):
        # First find the step of soln closest to what user typed
        curr_embedding = self.model.encode(user_state)

        min_dist = None
        min_idx = 0

        for i in range(len(self.sentence_embeddings)):
            sent_embedding = self.sentence_embeddings[i]
            sim = self.get_similarity(curr_embedding, sent_embedding)

            if min_dist is None or sim > min_dist:
                min_dist = sim
                min_idx = i

        return min_idx


if __name__ == '__main__':
    soln = open("solution.txt").read()
    tutor = Tutor(soln)

    user_state = "I know it's a geometric series, so the ratio should be log_2(x)/log_4(x). But I'm not sure what to do next"
    idx = tutor.get_user_state(user_state)
    print(idx)
    print(tutor.soln_sep[idx])
