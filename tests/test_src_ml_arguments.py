from src.ml.arguments import TrainScriptArguments, NotebookArguments


def test_training_arguments():
    args = TrainScriptArguments()
    assert type(args) == TrainScriptArguments


def test_notebook_arguments():
    args = NotebookArguments()
    assert type(args) == NotebookArguments
