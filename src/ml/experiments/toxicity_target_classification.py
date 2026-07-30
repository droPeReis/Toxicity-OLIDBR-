from experiments.toxic_comment_classification import ToxicCommentClassification


class ToxicityTargetClassification(ToxicCommentClassification):
    name = "toxicity-target-classification"

    labels = {0: "UNTARGETED", 1: "TARGETED INSULT"}
