import string
import unicodedata
import torch


ALPHANUM = f"{string.ascii_uppercase}{string.ascii_lowercase}{string.digits}"
# 62 == 10 + 26*2
ALPHA_NUM_PUNCT = f"` {ALPHANUM}'-\"/,.+_!#&():;?"
# 80 == 62 + 18


# -\'.ü!"#%&()*+,/:;?
Alphabets = {
    # '!#&():;?*%'
    "all": ALPHA_NUM_PUNCT,
    "iam_word": ALPHA_NUM_PUNCT,
    "iam_line": ALPHA_NUM_PUNCT,
    "cvl_word": ALPHA_NUM_PUNCT,
    "custom": ALPHA_NUM_PUNCT,
    # 'cvl_word': '` ABDEFGHILNPRSTUVWYZabcdefghiklmnopqrstuvwxyz\'-_159', # n_class: 52
    "rimes_word": f"` {ALPHANUM}%'-/Éàâçèéêëîïôùû",  # n_class: 81 == 62 + 19
}


class StrLabelConverter:
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_key, ignore_case=False):
        alphabet = Alphabets[alphabet_key]
        # print(alphabet)
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        self.dict = {char: i for i, char in enumerate(alphabet)}

    def encode(self, text, max_len=None):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if len(text) == 1:
            text = text[0]

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            return text

        length = []
        result = []
        results = []

        print("encode: ", text)

        for item in text:
            # item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
            results.append(result)
            result = []

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(text) for text in results], batch_first=True
        )
        lengths = torch.IntTensor(length)

        if max_len is not None and max_len > labels.size(-1):
            pad_labels = torch.zeros((labels.size(0), max_len)).long()
            pad_labels[:, : labels.size(-1)] = labels
            labels = pad_labels

        return labels, lengths

    def decode(self, t, length=None, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        def nonzero_count(x):
            return len(x.nonzero(as_tuple=False))

        if isinstance(t, list):
            t = torch.IntTensor(t)
            length = torch.IntTensor([len(t)])
        elif length is None:
            length = torch.IntTensor([nonzero_count(t)])

        if length.numel() == 1:
            length = length[0]
            assert (
                nonzero_count(t) == length
            ), f"{t} text with length: {nonzero_count(t)} does not match declared length: {length}"
            if raw:
                return "".join([self.alphabet[i] for i in t])

            char_list = []
            if t.dim() == 2:
                t = t[0]
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i]])
            return "".join(char_list)
        # batch mode
        assert (
            nonzero_count(t) == length.sum()
        ), f"texts with length: {nonzero_count(t)} does not match declared length: {length.sum()}"
        texts = [
            self.decode(t[i, : length[i]], torch.IntTensor([length[i]]), raw)
            for i in range(length.numel())
        ]
        return texts


def get_true_alphabet(name: str):
    tag = name.split("_", maxsplit=2)[:2]
    tag = "_".join(tag)
    return Alphabets[tag]


def get_lexicon(path, true_alphabet, max_length=20, ignore_case=True):
    words = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 2:
                    continue
                word = "".join(ch for ch in line if ch in true_alphabet)
                if len(word) != len(line) or len(word) >= max_length:
                    continue
                if ignore_case:
                    word = word.lower()
                words.append(word)
    except FileNotFoundError as e:
        print(e)
    return words


def word_capitalize(word):
    word = list(word)
    word[0] = (
        unicodedata.normalize("NFKD", word[0].upper())
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
    word = "".join(word)
    return word
