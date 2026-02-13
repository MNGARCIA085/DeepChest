import matplotlib.pyplot as plt



def build_loss_plot(history):
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    fig, ax = plt.subplots()

    ax.plot(train_loss, label="train_loss")
    if val_loss:
        ax.plot(val_loss, label="val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    return fig



def build_transfer_loss_plot(results):
    p1 = results["phase1"]["history"]
    p2 = results["phase2"]["history"]

    train1, val1 = p1["loss"], p1.get("val_loss", [])
    train2, val2 = p2["loss"], p2.get("val_loss", [])

    train_all = train1 + train2
    val_all = val1 + val2

    split_epoch = len(train1)

    fig, ax = plt.subplots()

    ax.plot(train_all, label="train_loss")
    if val_all:
        ax.plot(val_all, label="val_loss")

    ax.axvline(
        x=split_epoch - 0.5,
        linestyle="--",
        label="fine_tuning_start"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Feature Extraction → Fine Tuning")
    ax.legend()

    return fig



def build_precision_recall_plot(curves):
    fig, ax = plt.subplots(figsize=(7, 7))

    for class_name, data in curves.items():
        ax.plot(
            data["recall"],
            data["precision"],
            label=f"{class_name} (AP={data['ap']:.2f})"
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Multilabel)")
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True)

    return fig
