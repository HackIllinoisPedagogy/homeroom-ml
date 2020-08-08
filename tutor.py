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

if __name__ == '__main__':
    soln = open("solution.txt").read()
    tutor = Tutor(soln)
