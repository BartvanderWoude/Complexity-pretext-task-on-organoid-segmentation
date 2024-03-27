import os
import torch


class Logger:
    def __init__(self, type_training, task1, task2):
        self.type_training = type_training
        self.task1 = task1
        self.task2 = task2

        self.logs_path = "output/logs/"
        self.models_path = f"output/models/{type_training}/{task1}_{task2}/"

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        self.training = open(f"{self.logs_path}{type_training}_training_loss_{task1}_{task2}.csv", "w")
        self.training.write("fold,epoch,loss\n")

        self.validation = open(f"{self.logs_path}{type_training}_validation_loss_{task1}_{task2}.csv", "w")
        self.validation.write("fold,epoch,loss\n")

    def save_model(self, model: torch.nn.Module, fold):
        torch.save(model.state_dict(), "%sf%s_%s_%s_%s.pth" % (self.models_path,
                                                               str(fold),
                                                               self.type_training,
                                                               self.task1,
                                                               self.task2))

    def log_training_loss(self, fold, epoch, loss):
        if self.training.closed:
            raise ValueError("Logger is closed.")

        self.training.write("%s,%s,%s\n" % (str(fold),
                                            str(epoch),
                                            str(loss)))

    def log_validation_loss(self, fold, epoch, loss):
        if self.validation.closed:
            raise ValueError("Logger is closed.")

        self.validation.write("%s,%s,%s\n" % (str(fold),
                                              str(epoch),
                                              str(loss)))

    def close(self):
        self.training.close()
        self.validation.close()
