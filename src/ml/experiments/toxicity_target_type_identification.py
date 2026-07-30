from experiments.toxic_comment_classification import ToxicCommentClassification


class ToxicityTargetTypeIdentification(ToxicCommentClassification):
    name = "toxicity-target-type-identification"

    labels = {0: "INDIVIDUAL", 1: "GROUP", 2: "OTHER"}
