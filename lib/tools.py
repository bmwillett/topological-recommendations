import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


def get_batch(vocab, model, n_batches=1):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(10, 10))  # in inches
    texts = []
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        texts.append(plt.text(x,y,label,size=12))
                     # xytext=(5, 2),
                     # textcoords='offset points',
                     # ha='right',
                     # va='bottom',
                     # size=15))

    #     plt.savefig(filename)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.show()


    #
    #
    # together = [(0, 1.0, 0.4), (25, 1.0127692669427917, 0.41), (50, 1.016404709797609, 0.41),
    #             (75, 1.1043426359673716, 0.42), (100, 1.1610446924342996, 0.44), (125, 1.1685687930691457, 0.43),
    #             (150, 1.3486407784550272, 0.45), (250, 1.4013999168008104, 0.45)]
    # together.sort()
    #
    # text = [x for (x, y, z) in together]
    # eucs = [y for (x, y, z) in together]
    # covers = [z for (x, y, z) in together]
    #
    # p1 = plt.plot(eucs, covers, color="black", alpha=0.5)
    # texts = []
    # for x, y, s in zip(eucs, covers, text):
    #     texts.append(plt.text(x, y, s))
    #
    # plt.xlabel("Proportional Euclidean Distance")
    # plt.ylabel("Percentage Timewindows Attended")
    # plt.title("Test plot")
    # adjust_text(texts, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    # plt.show()