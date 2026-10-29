import torch
import torchmetrics


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor = None):
    """
    Compute balanced accuracy, confusion matrix, TPR/FPR (binary), and AUC.

    Returns:
        ba: balanced accuracy
        cm: confusion matrix
        tpr: true positive rate for binary classification, else None
        fpr: false positive rate for binary classification, else None
        auc: AUROC if y_prob is provided, else None
    """
    num_classes = int(y_true.max() + 1)
    cm = torchmetrics.functional.confusion_matrix(
        preds=y_pred,
        target=y_true,
        task="multiclass",
        num_classes=num_classes,
    )

    recalls = cm.diag() / (cm.sum(dim=1) + 1e-6)
    ba = recalls.nanmean()

    tpr = fpr = None
    if num_classes == 2:
        tn, fp, fn, tp = cm.flatten()
        tpr = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + tn + 1e-6)

    auc = None
    if y_prob is not None:
        if num_classes == 2:
            pos_prob = y_prob[:, 1]
            auc = torchmetrics.functional.auroc(preds=pos_prob, target=y_true, task="binary")
        else:
            auc = torchmetrics.functional.auroc(
                preds=y_prob,
                target=y_true,
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            )

    return ba, cm, tpr, fpr, auc


def cal_F1_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute weighted F1 score."""
    f1 = torchmetrics.F1Score(
        task="multiclass",
        num_classes=int(y_true.max() + 1),
        average="weighted",
    ).to(y_true.device)
    return f1(y_pred, y_true)
