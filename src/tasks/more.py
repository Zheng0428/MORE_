# coding=utf-8
# Copyleft 2019 project LXRT.
import sys
sys.path.append("./src")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6" 
import collections
from torch.utils.tensorboard import SummaryWriter


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from param import args
from tasks.more_model import MOREModel
from tasks.data_preprocessing.data_preprocessing import MiniGridDataset

HIDDEN_NUM = 20000
FINAL_LEN = 20
MAX_LENGTH = 1000

DataTuple = collections.namedtuple("DataTuple", 'dataset loader sampler')

def get_data_tuple(splits: str, bs:int, device, data_path, shuffle = False, drop_last = False) -> DataTuple:
    # /home/zhangge/ZTY_Adam/data/
    traj_dataset = MiniGridDataset(splits, path = data_path, max_length = MAX_LENGTH, device = device)
    # a = len(traj_dataset)
    traj_data_sampler = DistributedSampler(traj_dataset, shuffle=shuffle)
    traj_data_loader = DataLoader(             
        traj_dataset,
        batch_size = bs,
        sampler = traj_data_sampler,
        shuffle = shuffle,   #shuffle = Ture :Randomly generated idx
        pin_memory = True,   
        drop_last = drop_last,
        num_workers = 8
        )                         
    return DataTuple(dataset=traj_dataset, loader=traj_data_loader, sampler=traj_data_sampler)

def get_valid_tuple(splits: str, bs:int, device, data_path, shuffle = False, drop_last = False):
    # /home/zhangge/ZTY_Adam/data/
    traj_dataset = MiniGridDataset(splits, path = data_path, max_length = MAX_LENGTH, device = device)
    # a = len(traj_dataset)
    traj_data_loader = DataLoader(             
        traj_dataset,
        batch_size = bs,
        shuffle = shuffle,   #shuffle = Ture :Randomly generated idx
        pin_memory = True,   
        drop_last = drop_last,
        num_workers = 8
        )                         
    return traj_data_loader

class MORE:
    def __init__(self):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl')
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # GPU options
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, device=self.device, shuffle=False, drop_last=True, data_path = args.data_path
        )
        if args.valid != "":
            self.valid_tuple = get_valid_tuple(
                args.valid, bs=8, device=self.device,
                shuffle=False, drop_last=False,
                data_path = args.data_path
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = MOREModel() # Have already load the MORE_decoder weights
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)

        # Optimizer
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

    def train(self, train_tuple, eval_tuple):
        dset, loader, sampler = train_tuple
        log_dir = "./log/train_more"
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f"train{self.rank}"), flush_secs=10)
        # iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        best_valid = 100.
        print_interval = 0.25
        steps_per_epoch = len(loader)
        # valid_score = self.validate(eval_tuple)
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            for idx, (lxmert_out, rtg, traj_mask, timesteps) in enumerate(loader):
                self.model.train()
                self.optim.zero_grad()   #梯度归零
                lxmert_out, traj_mask, rtg, timesteps = lxmert_out.to(self.device), traj_mask.to(self.device), rtg.to(self.device), timesteps.to(self.device)
                output = self.model(lxmert_out, rtg, traj_mask, timesteps)
                loss = output.loss
                print (loss)
                writer.add_scalar("loss",loss,epoch * len(loader) + idx)
                loss.backward()  #反向传播
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)  #解决梯度爆炸的问题
                self.optim.step()  #参数更新

                if (idx + 1) % int(steps_per_epoch * print_interval) == 0 and self.rank == 0:
                    # 计算验证损失
                    valid_score = self.validate(eval_tuple)

                    # 打印日志
                    print(f"Epoch [{epoch+1}/{args.epochs}], Step [{idx+1}/{steps_per_epoch}], "
                        f"Training Loss: {loss.item():.5f}, Validation Loss: {valid_score:.5f}")
            # log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            log_str = ''
            if self.valid_tuple is not None and self.rank == 0:  # Do Validation on main process only
                # valid_score = self.validate(eval_tuple)
                if valid_score < best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            if self.rank == 0:
                print(log_str, end='')

                with open(self.output + "/log.log", 'a') as f:
                    f.write(log_str)
                    f.flush()
        writer.close()
        if self.rank == 0:
            self.save("LAST")

    # 验证函数
    def validate(self, loader):
        print ("begin valid!!")
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
             for i, (lxmert_out, rtg, traj_mask, timesteps) in enumerate(loader):
                lxmert_out, traj_mask, rtg, timesteps = lxmert_out.to(self.device), traj_mask.to(self.device), rtg.to(self.device), timesteps.to(self.device)
                outputs = self.model(lxmert_out, rtg, traj_mask, timesteps)
                loss = outputs.loss
                total_loss += loss.item()

        return total_loss / len(loader)
    
        
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
    more.train(
        more.train_tuple, 
        more.valid_tuple, 
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

