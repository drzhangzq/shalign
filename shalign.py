"""
Shaohua alignment for DNA sequence.
Version=1.1
2024-04-28
"""
from Bio import SeqIO
import numpy as np
import h5py
from math import inf
from queue import Queue
import re
#import HMM version 1.1
from HMM import HMM

#全局变量定义
g_lang="en" #language 'en':英文,'cn'：中文
g_h5file=''
g_line_len=20
g_nodes_per_level=1
g_sch_scal=0.1

class NewHMM(HMM):
    def __init__(self, dnalen):
        super().__init__(dnalen)

    def add_all_emmit_prob(self, prob_eps):
        # 为所与的M状态的发射射概率增加一个eps
        for state in range (self.dnalen):
            total=0.0
            for index in range(4):
                self.emission_probs[state][index] += prob_eps
                if self.emission_probs[state][index] < 0:
                    self.emission_probs[state][index] = 0
                total += self.emission_probs[state][index]
            self.emission_probs[state] /= total

class Node:
    def __init__(self, value, prob=None):
        self.value = value #偏移值，0：没有插入或删除变异，正整数：插入value个碱基，负整数：删除value个碱基
        self.prob = prob #观测序列出现的概率
        self.hiden_states=[]
        self.children = []
        self.father = None  # 添加父节点属性


def create_tree(hmm_obj:HMM, list_obv, line_len=60, max_size_level=1, searching_scal=0.1):
    """
    :param hmm_obj: 隐马尔科夫模型
    :param list_obv: 隐马尔科夫模型对应的观测序列
    :param line_len: 马尔科夫模型分解后的片段长度，默认是fasta文件的一行，60个碱基序列
    :param max_size_level: 搜索树每一层中保留的节点树，默认是1，即仅保留观测概率最大的解码方案
    :param searching_scal: 每次搜索时，马尔科夫模型分解后的片段长度的可变范围，当默认值为60时，可变长度范围为正负6。
    :return: 正常结束返回搜索树和叶子节点两个参数
    """

    root = Node(0, 1.0)  # 创建根结点
    queue = Queue()
    temp_queue = []
    queue.put((root, 0, 1.0))  # 0 表示所有父节点值之和，值的含义是引脚位置的偏移量, 1.0 表示出现概率位100%。
    obv_len = len(list_obv)-1
    dst = obv_len - hmm_obj.dnalen
    m = hmm_obj.dnalen//line_len
    seg_insert_rang = int(line_len * searching_scal)
    pian_min = min(-seg_insert_rang, dst - seg_insert_rang)
    pian_max = max(seg_insert_rang, dst + seg_insert_rang)

    for lv_m in range(m):
        level_size = queue.qsize()
        temp_lv = [[None for _ in range(pian_min, pian_max+1)] for _ in range(pian_min, pian_max+1)]
        seg_start = lv_m * line_len
        seg_end = seg_start + line_len - 1 if lv_m != m - 1 else hmm_obj.dnalen - 1  # 把最后一行剩余的碱基序列加到最后一行进行处理
        hmm_seg = create_hmm_seg(hmm_obj, seg_start, seg_end)
        debug = True
        for _ in range(level_size):
            node, parent_sum, parent_prob = queue.get()
            if node.value != inf:
                pianyi = parent_sum
                temp_index1=pianyi-pian_min
                if seg_end == hmm_obj.dnalen-1:
                    i_start= dst
                    i_end = dst
                else:
                    i_start= pian_min
                    i_end = pian_max
                for i in range(i_start, i_end+1):
                    #生成观测序列片段
                    seg_insert = i - pianyi
                    temp_index2 = i - pian_min
                    if (seg_end + pianyi + seg_insert + 1) > obv_len:
                        break
                    # 判断树状结构某层中的一个节点temp_lv[temp_index1][temp_index2]是否在之前的计算过程中计算过了。None表示没有被计算过
                    if temp_lv[temp_index1][temp_index2] is None:
                        seg = list_obv[seg_start + pianyi:seg_end + pianyi + seg_insert + 1]
                        hs=[]
                        if (seg_end + pianyi + seg_insert) < obv_len:
                            seg.append(hmm_obj.observations[-1])

                        k_max = hmm_seg.viterbi(seg, blockd=1)
                        prob_max = hmm_seg.cal_emt_prob(seg, k_max)

                        if prob_max == 0:
                            child_value = inf
                            prob_max = 0.0

                        else:
                            child_value = seg_insert
                            #生成隐藏状态序列hs

                            oi = 0
                            for si in k_max:
                                state = hmm_seg.states[si]
                                obv = seg[oi]
                                if state.startswith('M'):
                                    obv_idx = hmm_seg.observations.index(obv)
                                    oi += 1
                                    if hmm_seg.emission_probs[si][obv_idx] == max(hmm_seg.emission_probs[si]):
                                        pass
                                    else:
                                        hs.append(state)
                                elif state.startswith('I'):
                                    oi += 1
                                    hs.append(state)
                                else:
                                    hs.append(state)

                        # 在保存隐藏状态之前，需要检查隐藏状态是否同时出现了D和I，如果是，需要检查能否把D和I简化
                        # 初始化一个字典，用于保存隐藏状态
                        result_dict = {'M': [], 'D': [], 'I': [], 'E': []}
                        # 遍历列表中的每个字符串
                        for string in hs:
                            # 检查字符串的第一个字符是否为 M、D 或 I
                            if string[0] in ['M', 'D', 'I', 'E']:
                                # 将字符串的数字部分转换为整数
                                number = int(string[1:])
                                # 将整数添加到对应的键值列表中
                                result_dict[string[0]].append(number)
                        need_reduce = (len(result_dict['D']) > 0 and len(result_dict['I']) > 0)
                        if need_reduce:
                            # 第1步：生成参考序列 sequence_ref
                            sequence_ref = []
                            for i in range(hmm_seg.dnalen):
                                obv_id = np.argmax(hmm_seg.emission_probs[i])
                                obv_char = hmm_seg.observations[obv_id]
                                sequence_ref.append(obv_char)
                            sequence_ref.append(hmm_seg.observations[-1])

                            #第2步：对齐sequence_seg与sequence_ref
                            ocp_char = hmm_seg.observations[-2]
                            sequence_ocp = seg.copy()
                            if len(sequence_ocp) > len(sequence_ref):
                                #如果观测序列长于参考序列，先找出观测序列中插入的元素，通过对调对齐序列的参考序列与观测序列的角色
                                sequence_ref_copy = sequence_ref.copy()
                                sequence_aln_first=[]
                                sequence_aln_second=[]
                                sequence_aln=[]
                                sequence_aln_first = align_sequences(sequence_ref=sequence_ref, sequence_ocp=sequence_ocp,ocp_char=ocp_char)
                                sequence_aln_second = align_sequences(sequence_ref=sequence_ocp, sequence_ocp=sequence_ref_copy, ocp_char=ocp_char)
                                sequence_aln = align_sequences(sequence_ref=sequence_aln_second, sequence_ocp=sequence_ocp, ocp_char=ocp_char)
                            else:
                                sequence_aln = align_sequences(sequence_ref,sequence_ocp,ocp_char)
                            if len(sequence_aln) > 0:
                                #重新进行iterbi解码，但是不需要insert 'd'
                                k_max = hmm_seg.viterbi(sequence_aln, blockd=1, with_insertion_d=False)
                                prob_max = hmm_seg.cal_emt_prob(seg, k_max)
                                #debug=True
                                if prob_max > 0:
                                    #重新生存hs隐藏状态序列
                                    hs=[]
                                    # 生成隐藏状态序列
                                    oi = 0
                                    for si in k_max:
                                        state = hmm_seg.states[si]
                                        obv = seg[oi]
                                        if state.startswith('M'):
                                            obv_idx = hmm_seg.observations.index(obv)
                                            oi += 1
                                            if hmm_seg.emission_probs[si][obv_idx] == max(hmm_seg.emission_probs[si]):
                                                pass
                                            else:
                                                hs.append(state)
                                        elif state.startswith('I'):
                                            oi += 1
                                            hs.append(state)
                                        else:
                                            hs.append(state)


                        child = Node(child_value, prob_max)
                        child.hiden_states = hs
                        temp_lv[temp_index1][temp_index2] = child #将节点保存到temp_lv中，节点中不包含父节点信息。
                        child.father = node  # 设置子节点的父节点
                        temp_queue.append((child, parent_sum+child.value, parent_prob*child.prob))
                    else:
                        child_value = temp_lv[temp_index1][temp_index2].value
                        hs = temp_lv[temp_index1][temp_index2].hiden_states
                        prob_max= temp_lv[temp_index1][temp_index2].prob
                        child= Node(child_value, prob_max)
                        child.hiden_states = hs
                        child.father = node  # 设置子节点的父节点
                        temp_queue.append((child, parent_sum+child.value, parent_prob*child.prob))

        temp_queue.sort(key=lambda x: x[2], reverse=True)  # 按照最终prob的降序进行排序
        if max_size_level != None and max_size_level > 0 and len(temp_queue) > max_size_level:
            temp_queue = temp_queue[:max_size_level]  # 保留prob大的节点
        for nd, ps, psp in temp_queue:
            nd.father.children.append(nd)
            queue.put((nd, ps, psp))  # 将排序后的temp_queue中的节点重新放回原始队列queue中
        if lv_m < m-1: #如果不是叶子节点。
            temp_queue.clear()  # 清空临时队列

        # 输出进度条
        progress = (lv_m+1) / m
        progress_percent = int(progress * 100)
        bar = '>' * int(progress_percent*0.4) + '-' * (40 - int(progress_percent*0.4))
        print(f'\r[{bar}] {progress_percent}% ', end='', flush=True)
    print("\n")
    return root, temp_queue

def align_sequences(sequence_ref, sequence_ocp, ocp_char):
    # Create a matrix to store the alignment scores
    dp = [[0 for _ in range(len(sequence_ocp) + 1)] for _ in range(len(sequence_ref) + 1)]

    # Initialize the first row and column
    for i in range(len(sequence_ref) + 1):
        dp[i][0] = i
    for j in range(len(sequence_ocp) + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, len(sequence_ref) + 1):
        for j in range(1, len(sequence_ocp) + 1):
            if sequence_ref[i - 1] == sequence_ocp[j - 1] or (sequence_ref[i - 1] == ocp_char) or (
                    sequence_ocp[j - 1] == ocp_char):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # Backtrack to find the aligned sequence
    aligned_ocp = []
    i, j = len(sequence_ref), len(sequence_ocp)
    while i != 0 and j != 0:
        if sequence_ref[i - 1] == sequence_ocp[j - 1] or (sequence_ref[i - 1] == ocp_char) or (
                sequence_ocp[j - 1] == ocp_char):
            aligned_ocp.append(sequence_ocp[j - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] < dp[i][j - 1]:
            aligned_ocp.append(ocp_char)
            i -= 1
        else:
            aligned_ocp.append(sequence_ocp[j - 1])
            j -= 1

    # If there's a remaining gap at the beginning, fill it with ocp_char
    if i != 0:
        aligned_ocp += [ocp_char] * i

    return aligned_ocp[::-1]

def create_hmm_seg(hmm:HMM, seg_start, seg_end):
    hmm_seg = HMM(dnalen=seg_end - seg_start + 1)
    if seg_start > 0:
        start_prob = hmm.transition_probs[seg_start - 1].copy()
    else:
        start_prob = hmm.start_probs.copy()
    emission_prob = hmm.emission_probs[seg_start:seg_end + 1].copy()
    hmm_seg.start_probs[0] = start_prob[seg_start]  # 片段HMM中M0的概率
    hmm_seg.start_probs[hmm_seg.dnalen] = start_prob[seg_start + hmm.dnalen]
    hmm_seg.start_probs[hmm_seg.dnalen * 2 + 1] = start_prob[seg_start + hmm.dnalen * 2 + 1]

    for i in range(hmm_seg.dnalen):
        hmm_seg.emission_probs[i] = emission_prob[i]
    return hmm_seg

def add_number_to_string(input_string, increment):
    pattern = r'([A-Za-z]+)(\d+)'
    match = re.match(pattern, input_string)

    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        new_number = number + increment
        new_string = f'{prefix}{new_number}'
        return new_string
    else:
        return input_string

def get_number_in_string(input_string):
    pattern = r'([A-Za-z]+)(\d+)'
    match = re.match(pattern, input_string)
    number=-1

    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return number
    else:
        return number

def print_list_characters(input_list, line_len):
    input_string = ''.join(map(str, input_list))
    lines = [input_string[i:i+line_len] for i in range(0, len(input_string), line_len)]
    for line in lines:
        print(line)

def enquire_param():
    global g_line_len, g_nodes_per_level, g_sch_scal, g_lang
    prompt_len = f"Please enter the length for fragmenting DNA sequences (default is 20, current value is {g_line_len}): " if g_lang == "en" else f"请输入DNA序列片段化的长度（默认值为20,当前值为{g_line_len}）:"
    prompt_nodes = f"Please enter the maximum number of nodes to retain per level in the search tree (default is 1, current value is {g_nodes_per_level}):" if g_lang == "en" else f"请输入搜索树每层保留最大节点数（默认值为1,当前值为{g_nodes_per_level}）:"
    prompt_scal = f"Please enter the search variable range (0.01~0.50, default is 0.1, current value is {g_sch_scal}):" if g_lang == "en" else f"请输入搜索可变范围（0.01~0.50,默认值为0.1,当前值为{g_sch_scal}）:"
    line_len = input(prompt_len)
    if line_len.strip():
        g_line_len = int(line_len)

    nodes = input(prompt_nodes)
    if nodes.strip():
        g_nodes_per_level = int(nodes)

    scale = input(prompt_scal)
    if scale.strip():
        g_sch_scal = float(scale)

def get_hidden_states(node,ln_len):
    stack = []
    #node = leaf
    while node is not None:
        stack.append((node.hiden_states, node.prob))
        node = node.father
    # 从堆栈中读取数据并输出
    prob = 1.0
    pianyi = 0
    stack.pop()  # 删除根节点
    new_hs = []
    while stack:
        hs, probs = stack.pop()
        prob *= probs
        for s in hs:
            if s.startswith('I') or s.startswith('D') or s.startswith('M'):
                new_s = add_number_to_string(s, pianyi)
                new_hs.append(new_s)
            else:
                pass
        pianyi += ln_len
    return new_hs

def check_hidden_states(new_hs,HMM_obj:HMM, obs_seq):
    def process_elements(new_hs):
        q_d = []  # 创建一个空队列
        results = []  # 存储结果的列表
        last_d_num = None  # 记录最后一个D的数字
        last_i_num = None

        for element in new_hs:
            if element[0] == 'D':
                current_d_num = int(element[1:])  # 提取当前D后面的数字
                if not q_d or (last_d_num is not None and current_d_num == last_d_num + 1):
                    # 如果队列为空，或者当前D的数字是最后一个D数字加1（连续）
                    q_d.append(element)
                    last_d_num = current_d_num
                elif last_i_num == None:
                    # 如果不连续，且没有I，重置队列，并开始新的队列
                    q_d = [element]
                    last_d_num = current_d_num
                elif last_i_num is not None:
                    results.append(list(q_d))
                    q_d = []
                    last_d_num = None
                    last_i_num = None
            elif element[0] == 'M':
                if last_i_num == None:
                    # 如果遇到M，清空队列
                    q_d = []
                    last_d_num = None
                    last_i_num = None
                else:
                    results.append(list(q_d))
                    q_d = []
                    last_d_num = None
                    last_i_num = None
            elif element[0] == 'I':
                if q_d and q_d[-1][0] == 'D' or last_i_num == int(element[1:]):  # 确保队列中最后一个元素是D
                    q_d.append(element)
                    last_i_num = int(element[1:])
                elif q_d and last_d_num is not None and last_i_num is not None:
                    results.append(list(q_d))
                    q_d = []
                    last_d_num = None
                    last_i_num = None
            else:
                # 当取到的元素不是队列中的I时，检查队列是否有效并记录结果
                if q_d and q_d[-1][0] == 'I':
                    results.append(list(q_d))  # 添加到结果中
                q_d = []  # 清空队列
                last_d_num = None
                last_i_num = None

        # 检查最后一次迭代后队列中是否有未处理的有效数据
        if q_d and q_d[-1][0] == 'I':
            results.append(list(q_d))

        return results

    def extract_details(results):
        first_d_num = last_d_num = first_i_num = i_count = None
        first_i = None

        # 遍历结果列表寻找D和I
        for item in results:
            if item.startswith('D'):
                num = int(item[1:])
                if first_d_num is None:
                    first_d_num = num  # 第一个D后的数字
                last_d_num = num  # 最后一个D后的数字

            elif item.startswith('I'):
                num = int(item[1:])
                if first_i_num is None:
                    first_i_num = num  # 第一个I后的数字
                    first_i = item  # 第一个I

        # 计算特定I出现的次数
        if first_i:
            i_count = results.count(first_i)

        return first_d_num, last_d_num, first_i_num, i_count

    # 生成参考序列 sequence_ref
    sequence_ref = []
    for i in range(HMM_obj.dnalen):
        obv_id = np.argmax(HMM_obj.emission_probs[i])
        obv_char = HMM_obj.observations[obv_id]
        sequence_ref.append(obv_char)

    result = process_elements(new_hs)
    insertions = [int(item[1:]) for item in new_hs if item.startswith('I')]
    delets = [int(item[1:]) for item in new_hs if item.startswith('D')]
    remove_list=[]
    for i in range(len(result)):
        details = extract_details(result[i])
        d_start=details[0]
        d_end=details[1]+1
        pianyi=0
        for items in insertions:
            if items < d_end:
                pianyi += 1
            else:
                break
        for items in delets:
            if items < d_end:
                pianyi -= 1
            else:
                break
        i_start=d_end+pianyi
        i_end=details[2]+details[3] + pianyi

        seq0 = sequence_ref[d_start:d_end]
        seq1 = obs_seq[i_start:i_end]
        seq0_str = ''.join(seq0)
        seq1_str = ''.join(seq1)
        subsets_at = seq0_str.find(seq1_str)
        if subsets_at != -1:
            for j in range(subsets_at,subsets_at+details[3],1):
                remove_list.append(result[i][j])
            for j in range(-1,-(details[3]+1),-1):
                remove_list.append(result[i][j])
    #从new_hs中删除remove_list中的元素。
    if remove_list:
        for elem in remove_list:
            if elem in new_hs:
                new_hs.remove(elem)

def standoutput_aliangment(HMM_obj:HMM,sequence,hidden_states):
    #生成参考序列 sequence_ref
    sequence_ref=[]
    for i in range(HMM_obj.dnalen):
        obv_id=np.argmax(HMM_obj.emission_probs[i])
        obv_char=HMM_obj.observations[obv_id]
        sequence_ref.append(obv_char)
    #不输出结束符‘e’
    #sequence_ref.append(HMM_obj.observations[-1])
    lines_ref= [sequence_ref[i:i+60] for i in range(0, len(sequence_ref), 60)]

    index=0
    operations=[]
    current_ops=hidden_states[index]
    current_ops_pos=get_number_in_string(current_ops)
    pos = 0
    while pos < HMM_obj.dnalen + 1:
        if pos < current_ops_pos:
            operations.append('R')
        else:
            if current_ops.startswith('I'):
                operations.append('I')
                index+=1
                current_ops = hidden_states[index]
                current_ops_pos = get_number_in_string(current_ops)
                pos -= 1
            elif current_ops.startswith('D'):
                operations.append('D')
                index+=1
                current_ops = hidden_states[index]
                current_ops_pos = get_number_in_string(current_ops)
            elif current_ops.startswith('M'):
                operations.append('M')
                index+=1
                current_ops = hidden_states[index]
                current_ops_pos = get_number_in_string(current_ops)
        pos +=1
    operations.append('E')

    sequence_copy = sequence.copy()
    for i, op_char in enumerate(operations):
        if op_char == 'D':
            sequence_copy.insert(i, '-')

    rulers = "0    5    " * 6
    row = 0
    lines = f"{row:>12}"
    head_line = lines + rulers
    print(head_line)
    print("ref         "+"".join(lines_ref[row]))

    line1 = []  # 存储操作后的字符
    new_line = []  # 存储'I'操作生成的新行
    new_line_pos=0
    i=0
    prev_operation = None  # 用于追踪上一个操作

    for seq_char, op_char in zip(sequence_copy, operations):
        if i//60 > row:
            row = i //60
            lines = f"{row*6:>12}"
            head_line = lines + rulers
            print(head_line)
            print("ref         " + "".join(lines_ref[row]))
        if op_char == 'R':
            line1.append(seq_char)
            i+=1
        elif op_char == 'D':
            line1.append(seq_char)
            i+=1
        elif op_char == 'M':
            line1.append(seq_char.lower())
            i+=1
        elif op_char == 'I':
            if prev_operation != 'I':
                if new_line:
                    print(' '*(12+new_line_pos)+"".join(new_line))
                    new_line = []
                    new_line_pos=i%60
                    new_line.append(seq_char)
                else:
                    new_line_pos=i%60  # 记录起始位置
                    new_line.append(seq_char)
            else:
                new_line.append(seq_char)

        # 更新上一个操作
        prev_operation = op_char

        # 检查line1的长度是否达到60
        if len(line1) == 60:
            if new_line:
                print(' '*(12+new_line_pos)+"".join(new_line))
                new_line = []
            print(' '*12+"".join(line1))
            line1 = []

    # 打印剩余的字符
    if new_line:
        print(' '*(12+new_line_pos)+"".join(new_line))

    if line1:
        #line1.append(HMM_obj.observations[-1])
        print(' '*12+"".join(line1))
    return

def generate_model():
    global g_lang, g_h5file
    prompt_len = "Enter the length of the DNA sequence, excluding the terminator (e.g., for ACGe, the length is 3):" if g_lang == "en" else "请输入DNA序列的长度，不含结束符（如 ACGe 长度为3）："
    prompt_h5file = "Enter the filename for the DNA sequence model (e.g., model.hdf5):" if g_lang == "en" else "请输入DNA序列模型文件名（如model.hdf5）:"
    prompt_seqfile = "Enter the filename containing a group of DNA sequences in fasta format:" if g_lang == "en" else "请输入包含一组DNA序列的fasta文件名:"
    dnalen=input(prompt_len)
    dnalen=int(dnalen)
    seqfile=input(prompt_seqfile)
    h5file = input(prompt_h5file)
    hmm = NewHMM(dnalen)
    hmm.emission_prob_update(seqfile)
    hmm.add_all_emmit_prob(0.001)
    hmm.save_h5(h5file)
    hmm.print_model()
    g_h5file = h5file

    return

def demonstrate_model():
    # 实现模型演示的功能
    global g_h5file, g_line_len, g_nodes_per_level, g_sch_scal, g_lang
    #准备提示文本
    h5file_prompt = f"Enter the filename for the DNA sequence model (e.g., model.hdf5, current value is {g_h5file}):" if g_lang == "en" else f"请输入DNA序列模型文件名（如model.hdf5，当前值为{g_h5file}）:"
    #h5file=input(f"请输入DNA序列模型文件名（如model.hdf5,当前值为{g_h5file}）:")
    h5file=input(h5file_prompt)
    if h5file.strip():
        g_h5file=h5file
    else:
        h5file=g_h5file
    try:
        with h5py.File(h5file, 'r') as f:
            dnalen = f['dnalen'][()]
            hmm = HMM(dnalen)
            hmm.emission_probs = f['emission_matrix'][()]
            hmm.all_observations = f['all_observations'][()]
    except FileNotFoundError:
        prompt_notfound = f"File {h5file} not found. Please check the file path." if g_lang == "en" else f"文件 {h5file} 未找到，请检查文件路径是否正确。"
        print(prompt_notfound)
        return
    except Exception as e:
        prompt_fileerror = f"Error occurred while opening the file: {e}" if g_lang == "en" else f"打开文件时发生错误：{e}"
        print(prompt_fileerror)
        return

    prompt_times = "Enter the number of demonstrations (e.g., 2): " if g_lang == "en" else "请输入演示次数（如2）:"
    demontimes=input(prompt_times)
    if not demontimes.isdigit():
        number=2
    else:
        number = int(demontimes)
        if number <= 0:
            number=2

    enquire_param()

    for i in range(number):
        #排除掉没有任何变异的生成序列
        hiden_seq=['E']
        while (len(hiden_seq)<=5):
            obs_seq, hiden_seq, hs_idx_seq = hmm.generate_train_seq()

        #print(obs_seq)
        endchar = obs_seq.pop()
        print_list_characters(obs_seq, 60)
        obs_seq.append(endchar)
        print("Observation Probability=%e" % (hmm.cal_emt_prob(obs_seq, hs_idx_seq)))
        print(hiden_seq)

        #ln_len = g_line_len
        mytree, myleaf = create_tree(hmm, obs_seq, line_len=g_line_len, max_size_level=g_nodes_per_level, searching_scal=g_sch_scal)
        leaf, pianyi, prob = myleaf[0]
        new_hs = get_hidden_states(leaf, g_line_len)
        new_hs.append(hmm.states[-1])
        check_hidden_states(new_hs, hmm, obs_seq)
        standoutput_aliangment(hmm,obs_seq,new_hs)
        print("Decoded Probability=%e" % (prob))
        print(new_hs)

    return


def decode_sequence():
    #读取模型
    global g_h5file, g_line_len, g_nodes_per_level, g_sch_scal, g_lang
    prompt_h5file = f"Enter the filename for the DNA sequence model (e.g., model.hdf5, current value is {g_h5file}):" if g_lang == "en" else f"请输入DNA序列模型文件名（如model.hdf5，当前值为{g_h5file}）:"
    prompt_seqfile = "Enter the filename containing a group of DNA sequences in fasta format:" if g_lang == "en" else "请输入包含一组DNA序列的fasta文件名:"
    h5file = input(prompt_h5file)
    if h5file.strip():
        g_h5file = h5file
    else:
        h5file = g_h5file
    try:
        with h5py.File(h5file, 'r') as f:
            dnalen = f['dnalen'][()]
            hmm = HMM(dnalen)
            hmm.emission_probs = f['emission_matrix'][()]
            hmm.start_probs = f['start_matrix'][()]
            hmm.transition_probs = f['transition_matrix'][()]
            hmm.all_observations = f['all_observations'][()]
    except FileNotFoundError:
        prompt_notfound = f"File {h5file} not found. Please check the file path." if g_lang == "en" else f"文件 {h5file} 未找到，请检查文件路径是否正确。"
        print(prompt_notfound)
        return
    except Exception as e:
        prompt_fileerror = f"Error occurred while opening the file: {e}" if g_lang == "en" else f"打开文件时发生错误：{e}"
        print(prompt_fileerror)
        return

    seqfile = input(prompt_seqfile)
    seqs = []
    annos = []

    enquire_param()

    for fa in SeqIO.parse(seqfile, "fasta"):
        b_seq = list(fa.seq)
        if hmm.seq_is_dna(b_seq):
            b_seq.append(hmm.observations[-1])
            annos.append(fa.description)
            seqs.append(b_seq)
    #逐序列解码
    for i in range(len(seqs)):
        anno = annos[i]
        obs_seq = seqs[i]
        print('>'+anno)
        #print(obs_seq)
        #不输出结束符
        endchar=obs_seq.pop()
        print_list_characters(obs_seq, 60)
        obs_seq.append(endchar)
        #ln_len = g_line_len
        mytree, myleaf = create_tree(hmm, obs_seq, line_len=g_line_len, max_size_level=g_nodes_per_level, searching_scal=g_sch_scal)
        leaf, pianyi, prob = myleaf[0]
        new_hs = get_hidden_states(leaf, g_line_len)
        new_hs.append(hmm.states[-1])
        check_hidden_states(new_hs,hmm,obs_seq)
        standoutput_aliangment(hmm, obs_seq, new_hs)
        print("Decoded Probability=%e" % (prob))
        print(new_hs)

def print_menu():
    global g_lang
    menu_title="======= Program Menu =======" if g_lang=="en" else "======= 程序菜单 ======="
    menu1 = "1) Generate Model: Calculate the probability of each base appearing at each position in DNA sequences from a fasta file and save as hdf5." if g_lang == "en" else "1）模型生成：从fasta文件中统计DNA序列每个位置上碱基出现的概率，保存为hdf5格式。"
    menu2 = "2) Demonstrate Model: Generate DNA observation sequences based on the model and decode the corresponding hidden states." if g_lang == "en" else "2）解码演示：根据模型生成DNA观测序列，并解析观测序列对应的隐藏状态。"
    menu3 = "3) Decode Sequence: Read DNA observation sequences from a fasta file and decode the corresponding hidden states." if g_lang == "en" else "3）序列解码：从fasta文件中读取DNA观测序例，解析对应的隐藏状态。"
    menu4 = "4) Exit" if g_lang == "en" else "4）退出"
    print(menu_title)
    print(menu1)
    print(menu2)
    print(menu3)
    print(menu4)

def main():
    global g_lang
    prompt_choice = "Enter your choice (1-4):" if g_lang == "en" else "请输入您的选择（1-4）:"
    prompt_exit = "Program exit. Thank you for using!" if g_lang == "en" else "程序退出，感谢使用！"
    prompt_invalid = "Invalid choice. Please enter again." if g_lang == "en" else "无效的选择，请重新输入。"
    while True:
        print_menu()
        choice = input(prompt_choice)
        if choice == '1':
            generate_model()
        elif choice == '2':
            demonstrate_model()
        elif choice == '3':
            decode_sequence()
        elif choice == '4':
            print(prompt_exit)
            break
        else:
            print(prompt_invalid)

if __name__ == "__main__":
    main()
