# coding=utf-8
# Copyleft 2019 project LXRT.
import sys
sys.path.append("./src")
import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.more_model import MOREModel, CLModel
from tasks.data_preprocessing import MiniGridDataset
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

from lxrt.modeling import MLPModel
import torch.optim as optim
import torch.functional as F
from copy import deepcopy
HIDDEN_NUM = 20000
FINAL_LEN = 20

DataTuple = collections.namedtuple("DataTuple", 'dataset loader')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def get_data_tuple1(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    traj_dataset = MiniGridDataset(dataset_path='/home/zhangge/ZTY_Adam/MORE/data/more/minigrid_traj.pkl')
    traj_data_loader = DataLoader(             
        traj_dataset,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        drop_last=True
        )                         #shuffle = Ture :Randomly generated idx
    return DataTuple(dataset=traj_dataset, loader=traj_data_loader)

class MORE:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = MOREModel() # Have already load the MORE_decoder weights
        #build student model
        len_i = 36 + 20           #state's patch:36 and text's len:20
        len_o = FINAL_LEN
        self.student_model = MLPModel(len_i, HIDDEN_NUM, len_o)
        self.optimizer_student = optim.SGD(self.student_model.parameters(), 0.01)
        #build CL model for fitting GPT-2 input embedding
        self.cl_model = CLModel(
            lxrt_model = self.model.more_encoder,
            student_model = self.student_model,
            optimizer_student = self.optimizer_student,
            loss_fn=nn.KLDivLoss(),
            temp=20.0,
            device="cuda",
            log=False,
            logdir="./Experiments",
        )

        # Load MORE_Encoder weights
        if args.load_lxmert is not None:
            self.model.more_encoder.load(args.load_lxmert)
        # GPU options
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        if args.multiGPU:
            self.model.more_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)


    def train_cl(
        self, 
        train_tuple, 
        eval_tuple,
        epochs=10,
        plot_losses = True,
        save_model=True,
        save_model_pth="./models/student.pt",
    ):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        self.student_model.train()
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())
        best_valid = 0.

        best_acc = 0.0
        loss_arr = []
        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Training Student...")

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            correct = 0
            for i, (tmp, state, boxes, action, tmp) in iter_wrapper(enumerate(loader)):
                self.optim.zero_grad()   #梯度归零
                state, boxes = state.to(self.device), boxes.to(self.device)
                student_input = self.cl_model.student_input(state, boxes, action)
                student_out = self.student_model(student_input)
                teacher_input_id = self.model.tokenizer.batch_encode_plus(action, padding = 'max_length', truncation = True, return_tensors = 'pt', max_length = FINAL_LEN, add_special_tokens = True, return_attention_mask = True, return_token_type_ids = False).data['input_ids'].to(self.device)
                teacher_out = self.model.more_decoder.transformer.wte(teacher_input_id)
                loss = self.cl_model.calculate_kd_loss(student_out, teacher_out)
                loss.backward()  #反向传播
                nn.utils.clip_grad_norm_(self.student_model.parameters(), 5.)
                self.optim.step()  #参数更新

                #score, label = output.max(1)
               

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def train(self, train_tuple, eval_tuple, epoch_freeze):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.

        # Freeze the parameters of the encoder for the first few epochs
        for param in self.model.more_encoder.parameters():
            param.requires_grad = False
        # Unfreeze the parameters of the decoder
        for param in self.model.more_decoder.parameters():
            param.requires_grad = True

        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (timesteps, states, actions, rtg, traj_mask) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()   #梯度归零
                reward = torch.round(torch.rand(32, 1))
                target = 'turn left'
                states, boxes, rtg = states.to(self.device), boxes.to(self.device), rtg.to(self.device)
                
                output = self.model(states, boxes, actions, rtg, target, self.student_model)
                #assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(output, target)
                loss = loss * output.size(1)

                loss.backward()  #反向传播
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()  #参数更新

                score, label = output.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            #Unfreeze the parameters of the encoder after few epochs
            if epoch == epoch_freeze:                 
                for param in self.model.more_encoder.parameters():
                    param.requires_grad = True

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    more = MORE()

    # Load MORE model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        more.load(args.load)
    if args.cl_train:
        more.train_cl(
            more.train_tuple, 
            more.valid_tuple,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/student.pt",
        )
    else:
        more.train(
            more.train_tuple, 
            more.valid_tuple, 
            args.epoch_freeze
            ) # First run the training through
    # Test or Train
    # if args.test is not None:
    #    pass
    # else:
    #     print('Splits in Train data:', vqa.train_tuple.dataset.splits)
    #     if vqa.valid_tuple is not None:
    #         print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
    #         print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
    #     else:
    #         print("DO NOT USE VALIDATION")
    #     more.train(vqa.train_tuple, vqa.valid_tuple)


