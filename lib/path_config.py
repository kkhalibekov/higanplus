IMG_HEIGHT = 64
CHAR_WIDTH = IMG_HEIGHT // 2

data_roots = {"iam": "./data/iam/"}

data_paths = {
    "iam_word": {
        "trnval": f"trnvalset_words{IMG_HEIGHT}.hdf5",
        "test": f"testset_words{IMG_HEIGHT}.hdf5",
    },
    "iam_line": {
        "trnval": f"trnvalset_lines{IMG_HEIGHT}.hdf5",
        "test": f"testset_lines{IMG_HEIGHT}.hdf5",
    },
    "iam_word_org": {
        "trnval": f"trnvalset_words{IMG_HEIGHT}_OrgSz.hdf5",
        "test": f"testset_words{IMG_HEIGHT}_OrgSz.hdf5",
    },
}
