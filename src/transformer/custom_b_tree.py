class CustomBTree(object):

    def __init__(self):
        self.roots = []

    def find(self, query):
        found = []
        while len(query) > 1:
            # Get last word in token sequence
            last_word = query[-1]
            for node in self.roots:
                if node.id == last_word:
                    ret = node.find(query[:-1])
                    if ret is not None:
                        # Yay, found in tree
                        found.append(ret)

            # Cut start of query, i.e. make it one shorter from the start
            query = query[1:]

        # Prefer long sequence results over short sequence results
        best = found[0]

        return best

    def add(self, sequence, probas):
        last_word = sequence[-1]

        for node in self.roots:
            if node.id == last_word:
                node.add(sequence[:-1], probas)
                return

        self.roots.append(CustomBTreeNode(last_word, sequence[:-1], probas))


class CustomBTreeNode():
    def __init__(self, id, sequence, probas):
        self.id = id
        self.probas = []
        self.children = []
        if len(sequence) > 1:
            self.add(sequence, probas)
        else:
            self.probas = probas

    def find(self, query):
        last_word = query[-1]
        if len(query) == 1:
            return self.probas
        else:
            for node in self.children:
                if node.id == last_word:
                    return node.find(query[:-1])
            # Not found in tree
            return None

    def add(self, sequence, probas):
        if not sequence or not all(sequence):
            return
        last_word = sequence[-1]

        for i in range(len(self.children)):
            if self.children[i].id == last_word:
                self.children[i].add(sequence[:-1], probas)
                return

        self.children.append(CustomBTreeNode(last_word, sequence[:-1], probas))