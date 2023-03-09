# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import transformers
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU, MLPModel
import numpy as np

########################################
import torch
from torch.nn.parameter import Parameter
import math, time
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
########################################
# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20
HIDDEN_NUM = 10000
pad_id = 0
class MOREModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Build LXRT encoder
        self.more_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode = 'l'
        )
#        hid_dim = self.more_encoder.dim
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2") #load tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        transformers.logging.set_verbosity_error()
        self.more_decoder = transformers.GPT2LMHeadModel.from_pretrained('gpt2', num_hidden_layers = 6)


    def calculate_loss_and_accuracy(self, outputs, labels, device):
        """
        计算非pad_id的平均loss和准确率
        :param outputs:
        :param labels:
        :param device:
        :return:
        """
        logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
        # 用前n-1个token，预测出第n个token
        # 用第i个token的prediction_score用来预测第i+1个token。
        # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(device)

        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))

        _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

        # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
        not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def mlp(self, action, state, reward):
        action = self.mlp_model_a(action)
        state = self.mlp_model_s(state)
        reward = reward.unsqueeze(2).repeat(1, 1, 768)
        seq = torch.cat((action, state, reward),1)
        return seq

    def mix(self, action, state, reward):
        action = torch.flatten(action, 1, 2)
        state = torch.flatten(state, 1, 2)
        batch_size = action.shape[0]
        action = torch.cat([action, torch.ones(batch_size, 1).cuda()], dim=1)
        state = torch.cat([state, torch.ones(batch_size, 1).cuda()], dim=1)


        # 假设所设秩: R = 4, 期望融合后的特征维度: h = 768
        R, h = 4, self.gpt_input_dim - 1
        Wa = Parameter(torch.Tensor(R, action.shape[1], h)).cuda()
        Wa = torch.nn.init.xavier_normal_(Wa, gain=1.0).cuda()
        Ws = Parameter(torch.Tensor(R, state.shape[1], h)).cuda()
        Ws = torch.nn.init.xavier_normal_(Ws, gain=1.0).cuda()
        Wf = Parameter(torch.Tensor(1, R)).cuda()
        Wf = torch.nn.init.xavier_normal_(Wf, gain=1.0).cuda()
        bias = Parameter(torch.Tensor(1, h)).cuda()
        bias = torch.nn.init.xavier_normal_(bias, gain=1.0).cuda()

        # 分解后，并行提取各模态特征
        fusion_A = torch.matmul(action, Wa)
        fusion_S = torch.matmul(state, Ws)

        # 利用一个Linear再进行特征融合（融合R维度）
        funsion_AS = fusion_A * fusion_S
        funsion_AS = torch.matmul(Wf, funsion_AS.permute(1,0,2)).squeeze() + bias
        #fusion_AS = funsion_AS.detach().numpy()
        #最终输出的seq特征维度是（32，512）
        seq = torch.cat((funsion_AS, reward),1)
        return seq

    def forward(self, state, pos, action, reward, target, student_model):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param state: (b, o, f)
        :param pos:  (b, o, 4)
        :param action: (b,) Type -- list of string
        :param reward: (b, 1)
        :return: (b, 50257) 
        """
        texts = [
            'This is the first text.',
            'This is the second text.',
            'This is the third text.',
            'This is the fourth text.',
            'This is the fifth text.',
            'This is the sixth text.',
            'This is the seventh text.',
            'This is the eighth text.',
            'This is the ninth text.',
            'This is the tenth text.',
            'This is the eleventh text.',
            'This is the twelfth text.',
            'This is the thirteenth text.',
            'This is the fourteenth text.',
            'This is the fifteenth text.',
            'This is the sixteenth text.',
            'This is the seventeenth text.',
            'This is the eighteenth text.',
            'This is the nineteenth text.',
            'This is the twentieth text.',
            'This is the first text.',
            'This is the second text.',
            'This is the third text.',
            'This is the fourth text.',
            'This is the fifth text.',
            'This is the sixth text.',
            'This is the seventh text.',
            'This is the eighth text.',
            'This is the ninth text.',
            'This is the tenth text.',
            'This is the eleventh text.',
            'This is the twelfth text.',
        ]
        x = self.more_encoder(action, (state, pos))
        mix_seq = torch.cat((x[0], x[1]), 1)   #mix the output of the lxmert and the reward (numpy)
        output = student_model(mix_seq)
        target = self.tokenizer.batch_encode_plus(texts, padding = 'max_length', truncation = True, return_tensors = 'pt', max_length = MAX_VQA_LENGTH, add_special_tokens = True, return_attention_mask = True, return_token_type_ids = False)
        output = self.more_decoder(inputs_embeds = output, labels = target.data['input_ids'].cuda(), attention_mask = None)        #Be sure to pay attention to whether the input sequences are of the same length  #past_key_values = past 后面有时间可以加上
        #loss, accuray = self.calculate_loss_and_accuracy(outputs = output.logits, labels = action.data['input_ids'], device = 'cuda')
        return output

class CLModel:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        lxrt_model,
        student_model,
        optimizer_student,
        loss_fn=nn.KLDivLoss(),
        temp=20.0,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        self.lxrt_model = lxrt_model
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "Either an invalid device or CUDA is not available. Defaulting to CPU."
                )
                self.device = torch.device("cpu")

        self.student_model = student_model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.ce_fn = nn.CrossEntropyLoss().to(self.device)

    def student_input(self, state, boxes, action):
        x = self.lxrt_model(action, (state, boxes))
        output = torch.cat((x[0],x[1]),1)   #mix the output of the lxmert 
        return output

    def _train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pt",
    ):
        """
        Function to train student model - for internal use only.

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)

                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset

            _, epoch_val_acc = self._evaluate_model(self.student_model, verbose=True)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
                self.writer.add_scalar(
                    "Validation accuracy/Student", epoch_val_acc, epochs
                )

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pt",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self._train_student(epochs, plot_losses, save_model, save_model_pth)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset

        if verbose:
            print("-" * 80)
            print("Validation Accuracy: {}".format(accuracy))
        return outputs, accuracy

    def evaluate(self, teacher=False):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        _, accuracy = self._evaluate_model(model)

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass
    