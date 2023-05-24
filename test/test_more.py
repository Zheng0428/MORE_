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

DataTuple = collections.namedtuple("DataTuple", 'dataset loader')

def get_data_tuple(splits: str, bs:int, device, data_path, shuffle = False, drop_last = True) -> DataTuple:
    # /home/zhangge/ZTY_Adam/data/
    traj_dataset = MiniGridDataset(splits, path = data_path, max_length = 30, device = device)
    # a = len(traj_dataset)
    traj_data_loader = DataLoader(             
        traj_dataset,
        batch_size = bs,
        shuffle = shuffle,   #shuffle = Ture :Randomly generated idx
        pin_memory = True,   
        drop_last = drop_last,
        num_workers = 8
        )                         
    return DataTuple(dataset=traj_dataset, loader=traj_data_loader)

def get_valid_tuple(splits: str, bs:int, device, data_path, shuffle = False, drop_last = True):
    # /home/zhangge/ZTY_Adam/data/
    traj_dataset = MiniGridDataset(splits, path = data_path, max_length = 1000, device = device)
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
        # GPU options
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, device=self.device, shuffle=False, drop_last=True, data_path = args.data_path
        )
        if args.valid != "":
            self.valid_tuple = get_valid_tuple(
                args.valid, bs=16, device=self.device,
                shuffle=False, drop_last=True,
                data_path = args.data_path
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = MOREModel() # Have already load the MORE_decoder weights
        self.model = self.model.to(self.device)

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
        dset, loader = train_tuple
        log_dir = "./log/train_more"
        # iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        best_valid = 100.
        print_interval = 0.25
        steps_per_epoch = len(loader)
        # valid_score = self.validate(eval_tuple)
        for epoch in range(args.epochs):
            for idx, (lxmert_out, rtg, traj_mask, timesteps) in enumerate(loader):
                self.model.train()
                self.optim.zero_grad()   #梯度归零
                lxmert_out, traj_mask, rtg, timesteps = lxmert_out.to(self.device), traj_mask.to(self.device), rtg.to(self.device), timesteps.to(self.device)
                output = self.model(lxmert_out, rtg, traj_mask, timesteps)
                loss = output.loss
                loss.backward(retain_graph=True)  #反向传播
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)  #解决梯度爆炸的问题
                self.optim.step()  #参数更新

                if (idx + 1) % int(steps_per_epoch * print_interval) == 0:
                    # 计算验证损失
                    valid_score = self.validate(eval_tuple)
                    if self.valid_tuple is not None:
                        valid_score = self.validate(eval_tuple)

                        # 打印日志
                        print(f"Epoch [{epoch+1}/{args.epochs}], Step [{idx+1}/{steps_per_epoch}], "
                            f"Training Loss: {loss.item():.5f}, Validation Loss: {valid_score:.5f}")
                    else:
                        print(f"Epoch [{epoch+1}/{args.epochs}], Step [{idx+1}/{steps_per_epoch}], "
                            f"Training Loss: {loss.item():.5f}")
            # log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            log_str = ''
            if self.valid_tuple is not None:  # Do Validation on main process only
                # valid_score = self.validate(eval_tuple)
                if valid_score < best_valid:
                    best_valid = valid_score
                    # self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            else:
                log_str += "done"


            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
        # writer.close()

        # self.save("LAST")
    def validated(self):
        if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
        elif self.config.model_type == 'reward_conditioned':
            if self.config.game == 'Breakout':
                eval_return = self.get_returns(90)
            elif self.config.game == 'Seaquest':
                eval_return = self.get_returns(1150)
            elif self.config.game == 'Qbert':
                eval_return = self.get_returns(14000)
            elif self.config.game == 'Pong':
                eval_return = self.get_returns(20)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def get_returns(self, ret):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return
    
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

class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

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

