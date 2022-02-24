from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch


from model import CVRAE_next_action

class ActionForecastingDataset(Dataset):
    def __init__(self,dataset, split, obs_percent='obs_percent', train_or_test = 'train', ground_truth=True):

        """
        :param dataset: 'breakfasts' or '50salads'
        :param split: if dataset ==  'breakfasts', splits = 1-4, if '50salads', splits = 1-5
        :param obs_percent: percentage of entire sequence to observe .2 or .3
        :param norm_dur: [Boolean] whether to normalise action durartion
        :param ground_truth: [Boolean] whether to use groundtruth or mstcn annotations
        """
        self.train_or_test = train_or_test

        self.obs_percent = obs_percent

        self.act2ix, self.ix2act = read_mapping_dict(dataset)
        self.n_acts = len(self.act2ix)

        self.raw_sequences = read_files(dataset,split,train_or_test,ground_truth)

        # store duration of each action in list
        self.list_of_seqs_durations = []
        for seq in self.raw_sequences:
            for k, g in itertools.groupby(seq):
                duration = len(list(g))
                self.list_of_seqs_durations.append(duration)

        self.dur_mean = np.mean(np.array(self.list_of_seqs_durations))
        self.dur_std = np.std(np.array(self.list_of_seqs_durations))

        self.training_acts = []
        self.training_durs = []

        if train_or_test == 'train':
            for sequence in self.raw_sequences:
                acts,durs = self.get_action_and_duration_from_framewise_sequence(sequence)

                for i in range(2, len(acts)+1):
                    self.training_acts.append(acts[:i])
                    self.training_durs.append(durs[:i])

        #print(self.training_acts)
    def __len__(self):
        if self.train_or_test == 'train':
            return len(self.training_acts)
        elif self.train_or_test == 'test':
            return len(self.raw_sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train_or_test == 'train':
            return self.training_acts[idx], self.training_durs[idx]

        elif self.train_or_test == 'test':
            obs_seq, fut_seq = self.split_sequence_into_obs_and_future(self.raw_sequences[idx])
            obs_acts,obs_durs = self.get_action_and_duration_from_framewise_sequence(obs_seq)
            fut_acts, fut_durs = self.get_action_and_duration_from_framewise_sequence(fut_seq)
            return (obs_acts, obs_durs), (fut_acts, fut_durs)

    def get_action_and_duration_from_framewise_sequence(self, sequence):
        actions = []
        durations = []

        for act, g in itertools.groupby(sequence):
            actions.append(self.act2ix[act])
            durations.append(len(list(g)))

        return actions, durations

    def sequence_of_act_segments_to_tensor(self, act_segment):
        if self.task == 'dense_action':
            act_segment_tensor = torch.zeros(len(act_segment), len(self.act2ix) + 1)

            acts = torch.Tensor([self.act2ix[act[0]] for act in act_segment]).type(torch.long)
            acts_onehot = F.one_hot(acts, num_classes=len(self.act2ix))
            durations = torch.Tensor([act[1] for act in act_segment])
            for t in range(act_segment_tensor.size(0)):
                act_segment_tensor[t, :len(self.act2ix)] = acts_onehot[t]
                act_segment_tensor[t, -1] = durations[t]

        elif self.task == 'next_action' or self.task == 'next_action_seq':
            acts = torch.Tensor([self.act2ix[act] for act in act_segment]).type(torch.long)
            act_segment_tensor = F.one_hot(acts, num_classes=len(self.act2ix)).type(torch.float)

        return act_segment_tensor

    def split_sequence_into_obs_and_future(self, sequence):
        sample_length = len(sequence)
        obs_till = int(self.obs_percent * sample_length)

        obs = sequence[:obs_till]
        fut = sequence[obs_till:]

        return obs, fut

class NextActionDataset(Dataset):
    def __init__(self, train_or_test, dataset, split, n_obs='n_obs', ground_truth=True):
        """
        :param dataset: 'breakfasts' or '50salads'
        :param split: if dataset ==  'breakfasts', splits = 1-4, if '50salads', splits = 1-5
        :param n_obs: number of actions to observe
        :param ground_truth: [Boolean] whether to use groundtruth or mstcn annotations
        """
        self.n_obs = n_obs
        self.act2ix, self.ix2act = read_mapping_dict(dataset)
        self.n_acts = len(self.act2ix)

        self.raw_sequences = read_files(dataset, split, train_or_test, ground_truth)

        self.next_action_examples = []

        for sequence in self.raw_sequences:
            act_sequence = self.framewise_sequence_to_sequence_of_act_segments(sequence)
            training_examples_from_one_seq = generate_examples_from_one_seq(act_sequence,n_obs)

            self.next_action_examples.extend(training_examples_from_one_seq)

    def __len__(self):
        return len(self.next_action_examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.next_action_examples[idx]

    def framewise_sequence_to_sequence_of_act_segments(self, sequence):
        acts = []

        for act, g in itertools.groupby(sequence):
            acts.append(act)

        return torch.Tensor([self.act2ix[act] for act in acts]).type(torch.long)

    def sequence_of_act_segments_to_tensor(self, act_segment):
        acts = torch.Tensor([self.act2ix[act] for act in act_segment]).type(torch.long)
        act_segment_tensor = F.one_hot(acts, num_classes=len(self.act2ix)).type(torch.float)
        return act_segment_tensor

def generate_train_batch(data_batch):
    obs_act_batch = []
    obs_dur_batch = []

    target_act_batch = []
    target_dur_batch = []
    for acts, durs in data_batch:
        obs_acts = acts[:-1]
        target_act = acts[-1]

        obs_durs = durs[:-1]
        target_dur = durs[-1]

        obs_act_batch.append(obs_acts)
        obs_dur_batch.append(obs_durs)
        target_act_batch.append(target_act)
        target_dur_batch.append(target_dur)
    # observed_batch = pad_sequence(observed_batch)
    # future_batch = pad_sequence(target_batch)

    return [obs_act_batch,obs_dur_batch], [target_act_batch,target_dur_batch]


trainset = NextActionDataset('train',dataset='50salads', split=1, n_obs=3,ground_truth=True)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

model = CVRAE_next_action(trainset.n_acts,1,1,1)

x,y = next(iter(trainloader))

print(model(x))