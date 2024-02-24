from Bio import SeqIO
import numpy as np
import h5py

class HMM:
    #常数定义
    INITIPROB = 0.001
    INITDPROB = 0.001
    INITMPROB = 0.998
    IITRANSPROB = 0.2
    DDTRANSPROB = 0.2
    def __init__(self, dnalen):
        self.dnalen = int(dnalen)
        self.states = []
        for i in range(self.dnalen):
            sstat = 'M' + str(i)
            self.states.append(sstat)
        for i in range(self.dnalen + 1):
            sstat = 'I' + str(i)
            self.states.append(sstat)
        for i in range(self.dnalen):
            sstat = 'D' + str(i)
            self.states.append(sstat)
        self.states.append('E' + str(self.dnalen))
        self.state_num = len(self.states)
        self.all_observations = 0 #总的观测样本数，初始化为0
        # 生成self.pi
        self.start_probs = np.zeros(self.state_num)
        #初始概率，I0和D0均设置为
        self.start_probs[0] = 1.0 - self.INITIPROB - self.INITDPROB
        self.start_probs[self.dnalen] = self.INITIPROB
        self.start_probs[self.dnalen + self.dnalen + 1] = self.INITDPROB
        # 插入的dna序列与mute的没有区别，都是用ACGT表示，为了对齐dna序列，删除产生观测结果是必须出现在列表的到数第二位，这里定义为'd'，结束位产生观测结果'e'
        self.observations = ['A', 'C', 'G', 'T', 'd', 'e']
        self.transition_probs = np.zeros((len(self.states), len(self.states)))

        self.emission_probs = np.zeros((len(self.states), len(self.observations)))
        self.init_transition_probs()
        self.init_emission_probs()

    def init_transition_probs(self):
        #Mi->Ii+1或Di+1的概率初始化为最小值十万分之一
        for i in range(self.dnalen):
            if i != self.dnalen - 1:
                self.transition_probs[i][i + 1] = 1.0 - self.INITIPROB - self.INITDPROB
                self.transition_probs[i][i + self.dnalen + 1] = self.INITIPROB
                self.transition_probs[i][i + self.dnalen + 1 + self.dnalen + 1] = self.INITDPROB
            else:
                self.transition_probs[i][i * 3 + 4] = 1.0 - self.INITIPROB
                self.transition_probs[i][i + self.dnalen + 1] = self.INITIPROB
        for i in range(self.dnalen + 1):
            if i != self.dnalen:
                #设置Ii->Ii的概率，即在此处继续插入的概率
                self.transition_probs[self.dnalen + i][self.dnalen + i] = self.IITRANSPROB
                #设置Ii->Mi的概率，即在此处插入后不再插入而是定点变异的概率
                self.transition_probs[self.dnalen + i][i] = 1.0 - self.IITRANSPROB
            else:
                self.transition_probs[self.dnalen + i][self.dnalen + i] = self.IITRANSPROB
                self.transition_probs[self.dnalen + i][3 * self.dnalen + 1] = 1.0 - self.IITRANSPROB
        for i in range(self.dnalen):
            if i != self.dnalen - 1:
                self.transition_probs[i + 2 * self.dnalen + 1][i + 2 * self.dnalen + 2] = self.DDTRANSPROB
                self.transition_probs[i + 2 * self.dnalen + 1][i + 1] = 1.0 - self.DDTRANSPROB
            else:
                self.transition_probs[i + 2 * self.dnalen + 1][3 * i + 4] = 1.0
    def seq_is_dna(self, seq):
        for s in seq:
            res = False
            for s0 in self.observations:
                if s == s0:
                    res = True
            if res == False:
                return res
        return True
    def emission_prob_update(self, fastfile):
        seqs = []
        for fa in SeqIO.parse(fastfile, "fasta"):
            b_seq = np.array(list(fa.seq))

            if len(list(fa.seq)) == self.dnalen:
                if self.seq_is_dna(b_seq):
                    seqs.append(b_seq)
        new_observations = len(seqs)
        emiss = np.zeros((len(seqs[0]), 4))
        for i in range(len(seqs)):
            for j in range(len(seqs[0])):
                if seqs[i][j] == 'A':
                    emiss[j, 0] = emiss[j, 0] + 1
                if seqs[i][j] == 'C':
                    emiss[j, 1] = emiss[j, 1] + 1
                if seqs[i][j] == 'G':
                    emiss[j, 2] = emiss[j, 2] + 1
                if seqs[i][j] == 'T':
                    emiss[j, 3] = emiss[j, 3] + 1
        for i in range(self.dnalen):
            total_prob = 0.0
            for j in range(4):
                n_obs = self.emission_probs[i][j] * self.all_observations + emiss[i][j]
                self.emission_probs[i][j] = n_obs/(self.all_observations + new_observations)
                total_prob += self.emission_probs[i][j]
            if total_prob != 1.0:
                self.emission_probs[i] /= total_prob
        self.all_observations = self.all_observations + new_observations


    def init_emission_probs(self):
        for i in range(self.dnalen):
            j = i % 4
            self.emission_probs[i][j] = 0.25
        for i in range(self.dnalen + 1):
            self.emission_probs[i + self.dnalen][:4] = 0.25
        for i in range(self.dnalen):
            self.emission_probs[i + 2 * self.dnalen + 1][4] = 1.0
        self.emission_probs[3 * self.dnalen + 1][5] = 1.0
        self.all_observations = 0

    def forward(self, obs):
        N = len(obs)
        T = len(self.start_probs)
        alpha = np.zeros((N, T))
        index_obs = self.observations.index(obs[0])
        alpha[0, :] = self.start_probs * self.emission_probs[:, index_obs]
        for t in range(1, N):
            index_obst = self.observations.index(obs[t])
            alpha[t, :] = np.sum(alpha[t - 1, :] * self.transition_probs.T * self.emission_probs[:, index_obst], axis=1)
        return alpha

    def backward(slef, obs):
        N = slef.state_num  # 状态数
        T = len(obs)  # 观测序列长度
        beta = np.zeros((N, T))  # 后向概率矩阵
        beta[-1, -1] = 1  # 初始化为1
        for t in range(T - 2, -1, -1):  # 逆序遍历观测序列
            for i in range(N):
                # 计算后向概率
                index_obst = slef.observations.index(obs[t + 1])
                beta[i, t] = np.sum(beta[:, t + 1] * slef.emission_probs[:, index_obst] * slef.transition_probs[i, :])
        # 返回观测序列出现的概率
        index_obst = slef.observations.index(obs[0])
        return np.sum(slef.start_probs * slef.emission_probs[:, index_obst] * beta[:, 0])

    def calculate_prob(self, obs, forward=False):
        if forward:
            alpha = self.forward(obs)
            j = -1
            for i in range(len(obs)):
                p = np.sum(alpha[j, :])
                j = j - 1
                if p != 0.0:
                    break
            return np.sum(alpha[-2, :])
        else:
            return self.backward(obs)

    def cal_emt_prob(self, obs, hdn):
        """
        计算发射概率
        :param obs: 观测序列，list元素为char
        :param hdn: 观测序列对应的隐藏状态序列，list元素为int
        :return: 返回观测序列的概率float。
        """
        obs_len = len(obs)
        idx=0 #观测序例的索引
        state=self.states[hdn[idx]]
        prob=self.start_probs[hdn[idx]]
        for i in range(len(hdn)-1): # 遍历从第一个到倒数第二个隐藏状态
            if state.startswith('M'):
                obv_idx=self.observations.index(obs[idx])
                prob=prob*self.emission_probs[hdn[i]][obv_idx]
                idx += 1
                if idx > obs_len -1:
                    return 0.0
            elif state.startswith('I'):
                obv_idx = self.observations.index(obs[idx])
                prob = prob * self.emission_probs[hdn[i]][obv_idx]
                idx += 1
                if idx > obs_len -1:
                    return 0.0
            else:
                #prob = prob * 0.2 # ACGT D 五选一，概率=0.2
                pass
            next_state=self.states[hdn[i+1]]
            prob=prob*self.transition_probs[hdn[i]][hdn[i+1]]
            if prob == 0.0:
                break
            state=next_state
        return prob
    def mu_pin(self, segs): #segs是每段DNA序列中碱基的数量
        MAXN=512
        pin_num = self.dnalen//segs
        if pin_num > MAXN:
            pin_num = MAXN
            segs=self.dnalen//pin_num
        pin_i=[] # 引脚的位置
        pin_o=[] # 引脚位置对应的观测值
        prob_min=1.0 # 引脚处发射矩阵最小值
        prob_min_pos=None #引脚处发射矩阵最小值对应的引脚位置。
        for i in range(pin_num):
            prob_max = 0.0
            for j in range (segs):
                prob_idx=np.argmax(self.emission_probs[i*segs+j])
                prob=self.emission_probs[i*segs+j][prob_idx]
                if prob >= prob_max:
                    prob_max=prob
                    idx=i*segs+j
            if prob_max < prob_min:
                prob_min = prob_max
                prob_min_pos = idx
            pin_i.append(idx)
            pin_o.append(self.observations[np.argmax(self.emission_probs[idx])])
        if self.dnalen % segs !=0: #如果上面的循环结束后，DNA序例中还有剩余
            prob_max=0.0
            for j in range (segs*pin_num, self.dnalen):
                prob_idx=np.argmax(self.emission_probs[j])
                prob=self.emission_probs[j][prob_idx]
                if prob >= prob_max:
                    prob_max=prob
                    idx=j
                    if prob < prob_min:
                        prob_min = prob
                        prob_min_pos = idx
            pin_i.append(idx)
            pin_o.append(self.observations[np.argmax(self.emission_probs[idx])])


        pin_i.append(self.dnalen)
        pin_o.append(self.observations[-1])
        return pin_i, pin_o, prob_min, prob_min_pos


    def print_model(self):
        print("DNAlen:")
        print(self.dnalen)
        print("AllObservations:")
        print(self.all_observations)
        print("States:")
        print(self.states)
        print("Observations:")
        print(self.observations)
        print("Start Probabilities:")
        print(self.start_probs)
        print("Transition Probabilities:")
        print(self.transition_probs)
        print("Emission Probabilities:")
        print(self.emission_probs)

    def generate_obs_seq(self):
        """
        生成一个观测序列
        :return: 返回一个生成的观测序列，序列以列表形式保存，最后一个元素为‘e’。
        """
        obs_seq = []
        state = np.random.choice(self.states, p=self.start_probs)
        id = self.states.index(state)
        while not state.startswith('E'):
            if state.startswith('M'):
                obs = np.random.choice(self.observations, p=self.emission_probs[id])
            if state.startswith('D'):
                obs = None
                # obs = np.random.choice(self.observations, p=self.emission_probs[id])
            if state.startswith('I'):
                obs = np.random.choice(self.observations, p=self.emission_probs[id])
            if obs != None:
                obs_seq.append(obs)
            # state_seq = np.random.choice(self.state_num, p=self.start_probs)
            # state_seqs.append(state_seq)
            next_state = np.random.choice(self.states, p=self.transition_probs[id])
            state = next_state
            id = self.states.index(state)
        obs = np.random.choice(self.observations, p=self.emission_probs[id])
        obs_seq.append(obs)
        return (obs_seq)

    def generate_train_seq(self):
        """
        生成一个用于训练的数据集，包括观测序列和对应的隐藏序列
        :return: 观测序列(obs_seq)，隐藏序列(y_seq)，隐藏序列索引值(y_key_seq)。
        """
        obs_seq = []
        y_seq = []
        y_key_seq = []
        state = np.random.choice(self.states, p=self.start_probs)
        id = self.states.index(state)
        while not state.startswith('E'):
            if state.startswith('M'):
                obs = np.random.choice(self.observations, p=self.emission_probs[id])
                obv_idx=self.observations.index(obs)
                prob_max=max(self.emission_probs[id])
                if self.emission_probs[id][obv_idx] == prob_max:
                    obsy = None
                else:
                    obsy = state
            if state.startswith('D'):
                obs = None
                obsy = state
                # obs = np.random.choice(self.observations, p=self.emission_probs[id])
            if state.startswith('I'):
                obs = np.random.choice(self.observations, p=self.emission_probs[id])
                obsy = state
            if obs != None:
                obs_seq.append(obs)
            if obsy != None:
                y_seq.append(obsy)
            # state_seq = np.random.choice(self.state_num, p=self.start_probs)
            # state_seqs.append(state_seq)
            y_key_seq.append(id)
            next_state = np.random.choice(self.states, p=self.transition_probs[id])
            state = next_state
            id = self.states.index(state)
        obs = np.random.choice(self.observations, p=self.emission_probs[id])
        obs_seq.append(obs)
        y_seq.append(self.states[-1])
        y_key_seq.append(id)
        return obs_seq, y_seq, y_key_seq

    def save_h5(self,h5file):
        with h5py.File(h5file,'w') as  f:
            f.create_dataset('dnalen',data=self.dnalen)
            f.create_dataset('emission_matrix',data=self.emission_probs)
            f.create_dataset('start_matrix',data=self.start_probs)
            f.create_dataset('transition_matrix',data=self.transition_probs)
            f.create_dataset('all_observations',data=self.all_observations)

    def viterbi(self, obs, blockd=3):
        """
        使用Viterbi算法来解码隐马尔科夫模型。
        shalign使用改进的viterbi算法，当temp全为零时，就把观测序列加一个d,使用树型结构对插入位置进行搜索，直至解码成功或添加的d个数超过了阈值。
        obs: 观测序列，list，元素为char
        blockd: 每次插入的删除元素的个数，默认值为3
        """

        def create_next_states(current_states):
            next_states = []
            for i in range(len(current_states)):
                if current_states[i] != 0:
                    if i < self.dnalen - 1:
                        next_states.append(i + 1)
                        next_states.append(i + self.dnalen + 1)
                        next_states.append(i + self.dnalen + 1 + self.dnalen)
                    elif i == self.dnalen - 1:
                        next_states.append(i + self.dnalen + 1)
                        next_states.append(self.dnalen * 3 + 1)
                    elif i < self.dnalen * 2:
                        next_states.append(i - self.dnalen)
                        next_states.append(i)
                    elif i == self.dnalen * 2:
                        next_states.append(self.dnalen * 3 + 1)
                        next_states.append(i)
                    elif i < self.dnalen * 3:
                        next_states.append(i + 1)
                        next_states.append(i - self.dnalen * 2)
                    elif i == self.dnalen * 3:
                        next_states.append(self.dnalen * 3 + 1)
            res = set(next_states)

            return list(res)

        def insert_one_d(obslist, endp, symbd, datav, datab, repeat=1):
            #首先找到最后一个‘d’的位置。
            startp=-1
            res_obs = []
            res_viterbi = []
            res_backpointers = []
            res_max = 0.0
            #分别在startp+1到endp-1的位置上插入一个'd'
            for i in range(startp+1, endp+1):
                obs=obslist.copy()
                for j in range(repeat):
                    obs.insert(i,symbd)
                T = len(obs)
                N = len(self.states)
                # 初始化Viterbi矩阵和回溯矩阵。
                #viterbi = np.zeros((T, N))
                #backpointers = np.zeros((T, N), dtype=np.int32)
                viterbi = datav.copy()
                viterbi[i:,:]=0
                backpointers = datab.copy()
                backpointers[i:, :] = 0
                index_obs = self.observations.index(obs[i])
                # 初始化第i列的Viterbi值。
                if i!=0:
                    index_obs = self.observations.index(obs[i])
                    #生成下一个状态列表
                    current_states=viterbi[i-1]
                    next_states=create_next_states(current_states)
                    #for s in range(N):
                    for s in next_states:
                        # 计算每个状态的Viterbi值。
                        temp = viterbi[i - 1] * self.transition_probs[:, s] * self.emission_probs[s, index_obs]
                        # 选择Viterbi值最大的路径。
                        vts = np.max(temp)
                        viterbi[i, s] = vts
                        # 记录路径。
                        backpointers[i, s] = np.argmax(temp)
                else:
                    viterbi[i] = self.start_probs * self.emission_probs[:, index_obs]
                # 递推计算Viterbi矩阵和回溯矩阵。
                for t in range(i+1, T):
                    index_obs = self.observations.index(obs[t])
                    # 生成下一个状态列表
                    current_states = viterbi[t - 1]
                    next_states = create_next_states(current_states)
                    # for s in range(N):
                    for s in next_states:
                        # 计算每个状态的Viterbi值。
                        temp = viterbi[t - 1] * self.transition_probs[:, s] * self.emission_probs[s, index_obs]
                        # 选择Viterbi值最大的路径。
                        vts = np.max(temp)
                        viterbi[t, s] = vts
                        # 记录路径。
                        backpointers[t, s] = np.argmax(temp)
                    if np.max(viterbi[t]) == 0:
                        #print("t=%3d, i=%3d" %(t,i))
                        break
                tempmax = np.max(viterbi[endp])
                if tempmax > res_max:
                    res_max = tempmax
                    res_obs = obs.copy()
                    res_viterbi = viterbi.copy()
                    res_backpointers = backpointers.copy()
            #print (res_obs)
            if res_max:
                return (res_obs, res_viterbi, res_backpointers)
            else:
                return (obs, viterbi, backpointers)

        obsdel = self.observations[-2]
        T = len(obs)
        N = len(self.states)
        # 初始化Viterbi矩阵和回溯矩阵。
        viterbi = np.zeros((T, N))
        backpointers = np.zeros((T, N), dtype=np.int32)
        index_obs = self.observations.index(obs[0])
        # 初始化第一列的Viterbi值。
        viterbi[0] = self.start_probs * self.emission_probs[:, index_obs]

        # 递推计算Viterbi矩阵和回溯矩阵。
        t = 1
        while (t < T):
            index_obs = self.observations.index(obs[t])
            # 生成下一个状态列表
            current_states = viterbi[t - 1]
            next_states = create_next_states(current_states)
            # for s in range(N):
            for s in next_states:
                # 计算每个状态的Viterbi值。
                temp = viterbi[t - 1] * self.transition_probs[:, s] * self.emission_probs[s, index_obs]
                # 选择Viterbi值最大的路径。
                vts=np.max(temp)
                viterbi[t, s] = vts
                # 记录路径。
                backpointers[t, s] = np.argmax(temp)
            #如果返回的temp全零，则在位置t之前插入1个'd',直至viterbi的每一行都不全为零。
            while (np.max(viterbi[t]) == 0 and t < int(self.dnalen*2)): #增大self.dnalen后的系数可以提高解码成功率，当系数等于2.0时意味着把所有的观测序列都删除，一定可以通过插入生成任何需要的观测序列。
                #增加一行viterbi
                T = T + blockd
                viterbi = np.vstack([viterbi, np.zeros((blockd,N))])
                backpointers = np.vstack([backpointers, np.zeros((blockd, N),dtype=np.int32)])
                obscopy=obs.copy()
                obs, viterbi, backpointers = insert_one_d(obscopy, t, obsdel, viterbi, backpointers,repeat=blockd)
                t=t+blockd
                #print(t)
                #print(obs)
            t = t + 1
        # 根据回溯矩阵找到隐藏状态序列。
        hidden_states = np.zeros(T, dtype=np.int32)
        hidden_states[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            hidden_states[t] = backpointers[t + 1, hidden_states[t + 1]]

        return hidden_states
