from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrices(A_list, output_path):
    disp = [ConfusionMatrixDisplay(confusion_matrix=cm) for cm in A_list]
    # stack the confusion matrices horizontally
    fig, ax = plt.subplots(1, len(A_list), figsize=(20, 5))
    for i, d in enumerate(disp):
        d.plot(values_format='.2f')
        plt.savefig(output_path + f'{i}.pdf')

    plt.close()
