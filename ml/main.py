from ml.training.train_multiclassNN import train_multiclassNN
from ml.training.train_multiclassNN_HW_with_cross_validation import train_multiclassNN_HW_with_cross_validation
from ml.training.train_multiclassNN_with_cross_validation import train_multiclassNN_with_cross_validation
from ml.training.train_simpleNN import train_simpleNN
from ml.training.train_simpleNN_HW import train_simpleNN_HW
from ml.training.train_simpleNN_HW_with_cross_validation import train_simpleNN_HW_with_cross_validation
from ml.training.train_simpleNN_with_cross_validation import train_simpleNN_with_cross_validation


def main() -> None:
    # train_multiclassNN()
    # train_multiclassNN_with_cross_validation()
    # train_simpleNN()
    # train_simpleNN_with_cross_validation()
    # train_simpleNN_HW()
    # train_simpleNN_HW_with_cross_validation()
    train_multiclassNN_HW_with_cross_validation()

if __name__ == "__main__":
    main()