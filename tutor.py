import torch
import stanza

import re
import sys
import requests

from scipy import spatial
from sentence_transformers import SentenceTransformer

from collections import deque


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
        self.grammar_key = 'k94s3qewot7tswUd'
        self.grammar_url = 'https://api.textgears.com/check.php'


    def load_soln(self, soln):  # Given a solution, load it into the tutor
        self.soln = soln
        self.soln_sep = soln.split('.')

        new_soln_sep = []
        for s in self.soln_sep:
            if len(s.strip()) != 0:
                new_soln_sep.append(s)

        self.soln_sep = new_soln_sep

        self.sentence_embeddings = self.model.encode(self.soln_sep)
        self.reset_user_state()


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


    def get_hint(self, user_input):
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


    def get_possible_keywords(self, doc, cleaned_hint):
        possible = []
        cleaned_hint_possible = cleaned_hint.split(" ")

        # print(cleaned_hint, file=sys.stdout)

        for sent in doc.sentences:
            for word in sent.words:
                if ((word.upos in ["PROPN", "NOUN", "VERB"]) and
                    (word.deprel != "compound" and word.deprel != "appos") and
                    (word.head != 0) and
                    (not self.in_latex(cleaned_hint, word.text))):

                    possible.append(word)

        possible = [p for p in possible if p.text in cleaned_hint_possible]
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


    def get_under(self, node, graph):
        frontier = deque([node])
        visited = set([])

        while len(frontier) > 0:
            to_expand = frontier.popleft()
            if to_expand.deprel != "case":
                visited.add(to_expand)

            if to_expand in graph:
                for c in graph[to_expand]:
                    if c not in frontier and c not in visited:
                        frontier.append(c)

        out = list(visited)
        out.sort(key=lambda x: int(x.id))

        return out


    def in_latex(self, hint, token):
        if len(token) == 0:
            return False

        # Check if token is part of latex
        hint_latex_strings = re.findall(r"[$].*?[$]", hint)

        for l in hint_latex_strings:
            if token in l:
                return True

        return False


    def clean_curly_braces(self, hint, tokens):
        tokens_joined = " ".join(tokens)
        num_pre = 0
        num_post = 0

        for c in tokens_joined:
            if c == '{':
                num_post+=1
            elif c == '}':
                if num_post > 0:
                    num_post-=1
                else:
                    num_pre+=1

        tokens_joined = ("".join(["{" for i in range(num_pre)]) +
            tokens_joined + "".join(["}" for i in range(num_pre)]))

        return tokens_joined.split(" ")


    def clean_hint_latex_2(self, hint, tokens):
        tokens_joined = " ".join(tokens)

        print("tokens joined", file=sys.stdout)
        print(tokens_joined, file=sys.stdout)

        so_far = ""
        need_to_check = True

        prepend = False

        prev_closing = True    # Was the last $ closing

        for c in tokens_joined:
            if c == '$':    # Is this closing something before, or opening a new one?
                if need_to_check:
                    if self.in_latex(hint, so_far):     # It is closing
                        num_pre+=1
                        prepend = True
                    need_to_check = False
                    prev_closing = False
                else:
                    prev_closing = not prev_closing
            so_far += c

        out = ""
        if prepend:
            out += "$"

        out += tokens_joined

        if not prev_closing:
            out += "$"

        if self.in_latex(hint, out) and "$" not in out:
            out = "$" + out + "$"

        return self.clean_curly_braces(hint, out.split(" "))


    def clean_hint_latex(self, hint, tokens):
        l_active = False
        out_tokens = []

        for token in tokens:
            if self.in_latex(hint, token):
                # Update l_active
                if not l_active:    # We do not already have a starting $
                    if len(re.findall(r"[$].*?[$]", token)) == 0:   # Not self contained
                        if "$" in token:
                            l_active = True
                        else:
                            token = "$" + token
                else:   # We had already started latex
                    if "$" in token:
                        l_active = False

            out_tokens.append(token)

        # Need to check if the last token improperly terminated latex
        if self.in_latex(hint, out_tokens[-1]) and l_active:
            out_tokens[-1] = out_tokens[-1] + "$"

        return self.clean_curly_braces(hint, out_tokens)
        # return out_tokens


    def correct_grammar(self, sentence, sent_tokens=None):
        req = {"key": self.grammar_key, "text": sentence}
        errors = requests.get(self.grammar_url, req).json()['errors']
        errors = [err for err in errors if not self.in_latex(sentence, err['bad'])]

        # print(errors, file=sys.stdout)

        if sent_tokens is None:
            sent_tokens = sentence.split(' ')

        so_far = 0
        curr_token = 0
        curr_error = 0
        replace_with = None

        while True:
            if curr_error == len(errors):
                break

            if curr_token > len(sent_tokens):
                break

            t = sent_tokens[curr_token]
            so_far += len(t) + 1

            if replace_with is not None:
                sent_tokens[curr_token] = replace_with
                replace_with = None

            if so_far == errors[curr_error]['offset']:
                if len(errors[curr_error]['better']) > 0:
                    replace_with = errors[curr_error]['better'][0]
                else:
                    print("error", file=sys.stdout)
                    print(errors[curr_error], file=sys.stdout)
                curr_error+=1

            curr_token+=1

        return " ".join(sent_tokens)


    def get_sub_hint(self, hint_idx):
        # First try to do the 'smarter' dependency-tree based hint generation
        hint = self.soln_sep[hint_idx]

        # Clean hint of any latex
        cleaned_hint = re.sub("\$.*?\$", "", hint)

        doc = self.nlp(hint)
        possible_keywords = self.get_possible_keywords(doc, cleaned_hint)

        # print("Possible keywords", file=sys.stdout)
        # print(possible_keywords, file=sys.stdout)

        if self.num_in_state > len(possible_keywords) and self.try_tree:
            self.try_tree = False
            self.num_in_state = 1

        if len(possible_keywords) > 0 and self.try_tree:
            graph = self.make_graph(doc)
            hint_node = possible_keywords[self.num_in_state - 1]

            possible_words = [word.text for word in possible_keywords]
            word_embeddings = self.model.encode(possible_words)
            similarities = [self.get_similarity(w_embedding, self.sentence_embeddings[hint_idx]) for w_embedding in word_embeddings]

            ids = list(range(len(possible_words)))
            ids.sort(key=lambda x: -similarities[x])

            under_words = self.get_under(hint_node, graph)
            under_words_ids = [int(u.id) for u in under_words]

            min_id = min(under_words_ids)
            max_id = max(under_words_ids)

            # print(doc.sentences[0].words)
            to_include = doc.sentences[0].words[min_id: max_id+1]
            under = [u.text for u in to_include]

            # under = [u.text for u in self.get_under(hint_node, graph)]
            under = self.clean_hint_latex_2(hint, under)

            print("Dependency tree based")
            print(under, file=sys.stdout)

            return self.correct_grammar("Consider " + (" ".join(under)).lower() + ". How could this help?")

        # Find the most salient tokens of hint_idx
        tokens_sep = re.split(' |\n', self.soln_sep[hint_idx])
        tokens_sep = [t for t in tokens_sep if len(t) > 0]

        # tokens_sep = self.soln_sep[hint_idx].split(" ")
        salient_ids = self.find_salient(self.soln_sep[hint_idx],
            sentence_embedding=self.sentence_embeddings[hint_idx], tokens_sep=tokens_sep)

        # What is the length of the sub-hint we should give?
        len_hint = min(self.num_in_state, self.n_sub_hint) * len(salient_ids) / self.n_sub_hint
        to_include = [salient_ids[0]]
        s_id = 1

        while max(to_include) - min(to_include) + 1 < len_hint:
            to_include.append(salient_ids[s_id])
            s_id += 1

        including = tokens_sep[min(to_include): max(to_include) + 1]
        print("Saliency based")
        print(including, file=sys.stdout)

        including = self.clean_hint_latex(hint, including)

        return self.correct_grammar("Consider " + (" ".join(including)).lower() + ". How could this help?")


    def reset_user_state(self):
        self.num_in_state = -1
        self.curr_user_state = -1


if __name__ == '__main__':
    soln = open("solution.txt").read()
    tutor = Tutor(3, soln=soln)

    user_state = "I know it's a geometric series, so the ratio should be log_2(x)/log_4(x). But I'm not sure what to do next"
    for i in range(5):
        print(tutor.get_user_hint(user_state))
