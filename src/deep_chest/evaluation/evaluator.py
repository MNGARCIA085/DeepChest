from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# yhat to y_pred


class Evaluator:
    def __init__(self, threshold, labels):
        self.labels = labels
        self.threshold = threshold # a list of thresholds per label



    def accuracy_per_class(self, y_true, y_hat):
        return (y_true == y_hat).mean(axis=0)


    def recall_per_class(self, y_true, y_hat):        
        return recall_score(
            y_true,
            y_hat,
            average=None,
            zero_division=0
        )


    def prevalence(self, y_true):
        """
        Compute prevalence.
        Args:
            y_true (np.array): ground truth, size (n_examples)
        Returns:
            prevalence (float): prevalence of positive cases
        """
        return np.mean(y_true, axis=0)



    def specificity_per_class(self, y_true, y_hat):
        """
        as recall of the neg. class
        from sklearn.metrics import recall_score
        specificity_per_class = recall_score(
            y_true,
            y_hat,
            average=None,
            pos_label=0,
            zero_division=0
        )
        """
        tn = ((y_true == 0) & (y_hat == 0)).sum(axis=0)
        fp = ((y_true == 0) & (y_hat == 1)).sum(axis=0)

        return tn/(tn + fp + 1e-9)


    def ppv_per_class(self, y_true, y_hat):
        return precision_score(
            y_true,
            y_hat,
            average=None,
            zero_division=0
        )

    
    def npv_per_class(self, y_true, y_hat):
        tp = ((y_true == 1) & (y_hat == 1)).sum(axis=0)
        fp = ((y_true == 0) & (y_hat == 1)).sum(axis=0)
        tn = ((y_true == 0) & (y_hat == 0)).sum(axis=0)
        fn = ((y_true == 1) & (y_hat == 0)).sum(axis=0)

        return tn / (tn + fn + 1e-9)


    # auroc
    def auroc_per_class(self, y_true, y_pred_proba):
        return roc_auc_score(
            y_true,
            y_pred_proba,
            average=None
        )


    # auprc
    def auprc_per_class(self, y_true, y_pred_proba):
        scores = []
        for c in range(y_true.shape[1]):
            try:
                s = average_precision_score(y_true[:, c], y_pred_proba[:, c])
            except ValueError:
                s = np.nan  # class has only one label
            scores.append(s)
        return scores


    # F1-score
    def f1_per_class(self, y_true, y_hat):
        return f1_score(
            y_true,
            y_hat,
            average=None,
            zero_division=0
        )


    # main eval fn.
    def evaluate(self, y_true, y_pred_proba):        
        # Calculate thresholded labels ONCE
        y_hat = (y_pred_proba >= self.threshold).astype(int)
        
        # results
        per_class_metrics = {
            "accuracy": self.accuracy_per_class(y_true, y_hat),
            "recall": self.recall_per_class(y_true, y_hat),
            "sepcificity": self.specificity_per_class(y_true, y_hat),
            "f1": self.f1_per_class(y_true, y_hat),
            "prevalence": self.prevalence(y_true),
            "ppv_per_class": self.ppv_per_class(y_true, y_hat),
            "npv_per_class": self.npv_per_class(y_true, y_hat),
            "auroc": self.auroc_per_class(y_true, y_pred_proba), # Uses raw probs
            "auprc": self.auprc_per_class(y_true, y_pred_proba)
        }

        aggregate_metrics = self.aggregate_metrics(per_class_metrics, y_true, y_pred_proba)

        return per_class_metrics, aggregate_metrics


    # to calculate aggregate metrics
    def aggregate_metrics(self, results, y_true, y_pred_proba):
        agg = {}
        prevalence = np.array(results["prevalence"])

        for k, arr in results.items():
            arr = np.array(arr)

            if k == "prevalence":
                agg[f"{k}_mean"] = arr.mean()
                continue

            agg[f"{k}_macro"] = arr.mean()
            agg[f"{k}_std"] = arr.std()
            agg[f"{k}_weighted"] = np.average(arr, weights=prevalence)


        # Micro metrics
        agg["auroc_micro"] = roc_auc_score(
            y_true.ravel(), y_pred_proba.ravel()
        )

        agg["auprc_micro"] = average_precision_score(
            y_true.ravel(), y_pred_proba.ravel()
        )


        return agg


    # Precision-recall curves
    def get_precision_recall_curves(self, y_true, y_pred_proba):
        curves = {}

        for c, name in enumerate(self.labels):
            precision, recall, _ = precision_recall_curve(
                y_true[:, c],
                y_pred_proba[:, c]
            )

            ap = average_precision_score(
                y_true[:, c],
                y_pred_proba[:, c]
            )

            curves[name] = {
                "precision": precision,
                "recall": recall,
                "ap": float(ap)
            }

        return curves






    #------------Confidence intervales---------------------#
    def print_confidence_intervals(self, class_labels, statistics): # make it private
        df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
        for i in range(len(class_labels)):
            mean = statistics.mean(axis=1)[i]
            max_ = np.quantile(statistics, .95, axis=1)[i]
            min_ = np.quantile(statistics, .05, axis=1)[i]
            df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
        return df



    def bootstrap_auc(self, y, y_pred_proba, bootstraps=100, fold_size=1000): # check if its proba or labeled

        statistics = np.zeros((len(self.labels), bootstraps))

        for c in range(len(self.labels)):
            df = pd.DataFrame({
                "y": y[:, c],
                "proba": y_pred_proba[:, c]
            })

            df_pos = df[df.y == 1]
            df_neg = df[df.y == 0]
            prevalence = len(df_pos) / len(df)

            for i in range(bootstraps):
                pos_sample = df_pos.sample(
                    n=int(fold_size * prevalence), replace=True
                )
                neg_sample = df_neg.sample(
                    n=int(fold_size * (1 - prevalence)), replace=True
                )

                y_sample = np.concatenate([pos_sample.y, neg_sample.y])
                proba_sample = np.concatenate([pos_sample.proba, neg_sample.proba])

                statistics[c, i] = roc_auc_score(y_sample, proba_sample)

        return self.print_confidence_intervals(self.labels, statistics)



    # calibration curves
    def plot_calibration_curve(self, y_true, y_pred_proba):
        from sklearn.calibration import calibration_curve


        n = len(self.labels)
        cols = 4
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(5 * cols, 4 * rows))

        pred = y_pred_proba
        print(pred[:, 1].min(), pred[:, 1].max(), pred[:, 1].mean()) # 1 is i


        for i in range(n):
            plt.subplot(rows, cols, i + 1)

            if y_true[:, i].sum() == 0:
                plt.title(f"{self.labels[i]} (no positives)")
                plt.axis("off")
                continue

            frac_pos, mean_pred = calibration_curve(
                y_true[:, i],
                y_pred_proba[:, i],
                n_bins=20,
                strategy="quantile"
            )

            plt.plot([0, 1], [0, 1], "--", alpha=0.7)
            plt.plot(mean_pred, frac_pos, marker=".")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed frequency")
            plt.title(self.labels[i])

        plt.tight_layout()
        plt.show()



    """
    def plot_calibration_curve(self, y, pred):
        from sklearn.calibration import calibration_curve
        plt.figure(figsize=(20, 20))
        for i in range(len(self.labels)):
            plt.subplot(4, 4, i + 1)
            fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
            plt.xlabel("Predicted Value")
            plt.ylabel("Fraction of Positives")
            plt.title(self.labels[i])
        plt.tight_layout()
        plt.show()
    """
