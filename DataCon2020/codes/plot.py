from sklearn import metrics
import matplotlib.pyplot as plt


def plot_roc_curve(test_label, y_pred, *, model_name, save=True):
    """Calculate the AUC value of the model
       and drawing.
    :@param test_label: the actual label of the test set.
    :type test_label: the K dimension np.array.
    :@param y_pred: the predictive label of the model.
    :type y_pred: the K dimension np.array.
    :@param model_name: name of the model.
    :type model_name: str.
    :@param save: control the saving of images.
    :type save: bool.
    """

    font = {"color": "darkred", "size": 13, "family": "serif"}

    # calculate auc value
    fpr, tpr, _ = metrics.roc_curve(test_label, y_pred)
    auc = metrics.roc_auc_score(test_label, y_pred)

    # draw a roc curve
    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} AUC = {auc:.5f}",
            color="steelblue",
            rasterized=True,
            linewidth=2,
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontdict=font)
        ax.set_ylabel("True Positive Rate", fontdict=font)
        ax.set_title("ROC curve", fontdict=font)
        ax.legend(loc="lower right")
        ax.tick_params(axis="both")
        plt.tight_layout()
        if save:
            fig.savefig(f"{model_name}_auc_curve.pdf")


def plot_multiple_roc_curve(
    test_label_array,
    y_pred_array,
    model_name_list,
    data_volume_list,
    *,
    col,
    width,
    height,
    save=True,
):
    """Calculate the AUC value of the multiple model
       and drawing.
    :@param test_label_array: the actual label array of the test set.
    :type test_label_array: the MxK dimension np.array or list.
    :@param y_pred_array: the predictive label array of the model.
    :type y_pred: the MxK dimension np.array or list.
    :@param model_name_list: name list of the multiple model.
    :type model_name: list[str].
    :@param data_volume_list: the sample number of each training is listed.
    :type data_volume_list: list.
    :@param col: control the number of subgraphs.
    :type col: int.
    :@param width: the total width of the canvas.
    :type width: int.
    :@param height: the total height of the canvas.
    :type height: int.
    :@param save: control the saving of images.
    :type save: bool.
    """

    font = {"color": "#392f41", "size": 11, "family": "serif"}

    # calculate {tpr fpr auc} value and save as a list
    fpr_list, tpr_list, auc_list = [], [], []
    for test_label, y_pred in zip(test_label_array, y_pred_array):
        fpr, tpr, _ = metrics.roc_curve(test_label, y_pred)
        auc = metrics.roc_auc_score(test_label, y_pred)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    # calculate the number of rows in a subgraph
    if len(auc_list) % col:
        row = len(auc_list) // col + 1
    else:
        row = len(auc_list) // col

    with plt.style.context("bmh"):
        fig, axs = plt.subplots(row, col, figsize=(width, height))
        # while row or col is 1, add new dimension
        if row == 1 or col == 1:
            axs = axs[:, np.newaxis]
        axs = [i for ax in axs for i in ax]  # modify the dimensions of axs
        for ax, fpr, tpr, model_name, auc, volume in zip(
            axs, fpr_list, tpr_list, model_name_list, auc_list, data_volume_list
        ):

            ax.plot(
                fpr,
                tpr,
                label=f"{model_name} AUC = {auc:.5f}",
                color="steelblue",
                rasterized=True,
                linewidth=2,
            )

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate", fontdict=font)
            ax.set_ylabel("True Positive Rate", fontdict=font)
            ax.set_title(f"ROC curve (Data volume {volume})", fontdict=font)
            ax.legend(loc="lower right")
            ax.tick_params(axis="both")
            plt.tight_layout()

        if save:
            fig.savefig("multiple_auc_curve.pdf")
