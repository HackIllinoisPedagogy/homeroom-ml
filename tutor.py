import torch
import stanza

import re
import sys
import requests

from scipy import spatial
from sentence_transformers import SentenceTransformer
from question_generation.pipelines import pipeline
from nltk.metrics.distance import edit_distance

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

        self.question_generator = pipeline("e2e-qg")
        self.possible_questions = []


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
            if token in l.lower():
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

        tokens_joined = ("".join(["{ " for i in range(num_pre)]) +
            tokens_joined + "".join([" }" for i in range(num_pre)]))

        return tokens_joined.split(" ")


    def clean_slashes_latex(self, tokens):
        out = []
        for i in range(len(tokens)):
            t = tokens[i]
            if "frac{" in t:
                frac_pos = t.find("frac{")
                if frac_pos == 0 or t[frac_pos - 1] != "\\":
                    if len(out) > 0 and out[-1] == "\\":
                        out.pop()
                    t = "\\" + t
            out.append(t)
        return out


    def clean_hint_latex(self, hint, tokens):
        tokens = self.clean_curly_braces(hint, tokens)
        tokens = self.clean_slashes_latex(tokens)
        tokens_joined = " ".join(tokens)

        # print(tokens_joined, file=sys.stdout)

        so_far = ""
        need_to_check = True

        prepend = False

        prev_closing = True    # Was the last $ closing

        for c in tokens_joined:
            if c == '$':    # Is this closing something before, or opening a new one?
                if need_to_check:
                    if self.in_latex(hint, so_far.split(" ")[0]):     # It is closing
                        prepend = True
                    else:
                        prev_closing = False
                    need_to_check = False
                else:
                    prev_closing = not prev_closing
            so_far += c

        out = ""
        if prepend:
            out += "$ "

        out += tokens_joined

        if not prev_closing:
            out += " $"

        if self.in_latex(hint, out) and "$" not in out:
            out = "$ " + out + " $"

        return out.split(' ')


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

            if curr_token == len(sent_tokens):
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


    def model_questions(self, hint_idx):
        hint = self.soln_sep[hint_idx]

        poss_questions = self.question_generator(hint)
        processed_questions = []
        for question in poss_questions:
            cleaned_latex = " ".join(self.clean_hint_latex(hint, question.split()))
            processed_questions.append(self.fix_question(cleaned_latex, hint))
        poss_questions = list(set(processed_questions))
        # print(poss_questions, file=sys.stdout)
        return poss_questions


    def fix_question(self, question, full_hint):
        non_latex = re.split('[$].*?[$]', question)
        latex = re.findall('[$].*?[$]', question)
        hint_latex = re.findall('[$].*?[$]', full_hint)

        m = None
        m_idx = 0

        for i in range(len(latex)):
            l1 = latex[i]
            for j in range(len(hint_latex)):
                l2 = hint_latex[j]
                dist = edit_distance(l1, l2)

                if m is None or dist < m:
                    m = dist
                    m_idx = j

            if m < 8:   # Very low edit distance, probably the token
                latex[i] = hint_latex[m_idx]
                m = None

        curr = 0
        out = []

        while True:
            if curr % 2 == 0:
                if curr/2 > len(non_latex)-1:
                    break
                if len(non_latex[curr//2]) > 0:
                    out.append(non_latex[curr//2])
            else:
                if (curr-1)/2 > len(latex)-1:
                    break
                out.append(latex[(curr-1)//2])
            curr+=1

        return " ".join(out)


    def dep_tree_questions(self, hint_idx):
        hint = self.soln_sep[hint_idx]

        cleaned_hint = re.sub("\$.*?\$", "", hint)
        out = []

        doc = self.nlp(hint)
        possible_keywords = self.get_possible_keywords(doc, cleaned_hint)

        print("Possible keywords", file=sys.stdout)
        print(possible_keywords)

        graph = self.make_graph(doc)

        for hint_node in possible_keywords:
            under_words = self.get_under(hint_node, graph)
            under_words_ids = [int(u.id) for u in under_words]

            min_id = min(under_words_ids)
            max_id = max(under_words_ids)

            # print(doc.sentences[0].words)
            to_include = doc.sentences[0].words[min_id: max_id+1]
            under = [u.text for u in to_include]

            extracted_hint = (" ".join(under)).lower()

            print("extracted_hint")
            print(extracted_hint)

            questions = ["Consider " + " ".join(self.clean_hint_latex(hint, extracted_hint.split(" "))) + ". How could this help?"]
            # questions = [" ".join(self.clean_hint_latex(hint, q.split(" "))) for q in questions]

            out += questions

        return out

    def get_saliency_questions(self, hint_idx):
        out = []
        hint = self.soln_sep[hint_idx]

        # Find the most salient tokens of hint_idx
        tokens_sep = re.split(' |\n', self.soln_sep[hint_idx])
        tokens_sep = [t for t in tokens_sep if len(t) > 0]

        # tokens_sep = self.soln_sep[hint_idx].split(" ")
        salient_ids = self.find_salient(self.soln_sep[hint_idx],
            sentence_embedding=self.sentence_embeddings[hint_idx], tokens_sep=tokens_sep)

        # What is the length of the sub-hint we should give?
        for i in range(1, self.n_sub_hint + 1):
            len_hint = i * len(salient_ids) / self.n_sub_hint
            to_include = [salient_ids[0]]
            s_id = 1

            while max(to_include) - min(to_include) + 1 < len_hint:
                to_include.append(salient_ids[s_id])
                s_id += 1

            including = tokens_sep[min(to_include): max(to_include) + 1]

            extracted_hint = (" ".join(including)).lower()

            # print("extracted_hint")
            # print(extracted_hint)

            questions = ["Consider " + " ".join(self.clean_hint_latex(hint, extracted_hint.split(" "))) + ". How could this help?"]
            # questions = [" ".join(self.clean_hint_latex(hint, q.split(" "))) for q in questions]

            out += questions

        return out

    def get_sub_hint(self, hint_idx):
        if self.num_in_state == 1:

            model_questions = self.model_questions(hint_idx)
            dep_tree_questions = self.dep_tree_questions(hint_idx)
            saliency_questions = self.get_saliency_questions(hint_idx)

            print(len(model_questions), file=sys.stdout)
            print(len(dep_tree_questions), file=sys.stdout)
            print(len(saliency_questions), file=sys.stdout)

            self.possible_questions = (model_questions +
                dep_tree_questions + saliency_questions)

            print(self.possible_questions)

        idx = min(len(self.possible_questions)-1, self.num_in_state-1)
        return self.possible_questions[idx]


    def reset_user_state(self):
        self.num_in_state = -1
        self.curr_user_state = -1


if __name__ == '__main__':
    soln = open("solution.txt").read()
    tutor = Tutor(3, soln=soln)

    user_state = "I know it's a geometric series, so the ratio should be log_2(x)/log_4(x). But I'm not sure what to do next"
    for i in range(5):
        print(tutor.get_user_hint(user_state))
