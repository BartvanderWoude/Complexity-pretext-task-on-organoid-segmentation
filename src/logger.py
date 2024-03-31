import os
import torch


class Logger:
    def __init__(self, type_training, task1, task2, evaluating=False):
        self.type_training = type_training
        self.task1 = task1
        self.task2 = task2

        self.logs_path = "output/logs/"
        self.models_path = f"output/models/{type_training}/{task1}_{task2}/"
        self.evaluation_path = "output/evaluation/"

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        if not os.path.exists(self.evaluation_path):
            os.makedirs(self.evaluation_path)

        if evaluating:
            evaluation_file_path = f"{self.evaluation_path}{type_training}_evaluation.csv"
            if not os.path.exists(evaluation_file_path):
                self.evaluation = open(evaluation_file_path, "w")
                if type_training == "pretext":
                    self.evaluation.write("task1,task2,fold,psnr\n")
                elif type_training == "downstream":
                    self.evaluation.write("task1,task2,fold,f1\n")
            else:
                self.evaluation = open(f"{self.evaluation_path}{type_training}_evaluation.csv", "a")
            return

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

    def log_evaluation(self, task1, task2, fold, f1):
        if self.evaluation.closed:
            raise ValueError("Logger is closed.")

        self.evaluation.write("%s,%s,%s,%s\n" % (task1,
                                                 task2,
                                                 str(fold),
                                                 str(f1)))

    def close(self):
        self.training.close()
        self.validation.close()
