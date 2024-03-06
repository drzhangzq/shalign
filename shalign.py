from Bio import SeqIO
import numpy as np
import h5py
from math import inf
from queue import Queue
import re
from HMM import HMM

#全局变量定义
g_lang="en" #language 'en':英文,'cn'：中文
g_h5file=''
g_line_len=60
g_nodes_per_level=1
g_sch_scal=0.1

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
    queue.put((root, 0, 1.0))  # 0 表示所有父节点值之和，值的含义是引脚位置的偏移量。
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
                    if temp_lv[temp_index1][temp_index2] is None:
                        seg = list_obv[seg_start + pianyi:seg_end + pianyi + seg_insert + 1]
                        if (seg_end + pianyi + seg_insert) < obv_len:
                            seg.append(hmm_obj.observations[-1])
                        hs = []
                        k_max = []
                        if seg_insert < 0:
                            prob_max = 0.0
                            for j in range(1, -seg_insert + 1):
                                k = hmm_seg.viterbi(seg, blockd=j)
                                prob = hmm_seg.cal_emt_prob(seg, k)
                                if prob > prob_max:
                                    prob_max = prob
                                    k_max = k
                                    break

                        else:
                            k_max = hmm_seg.viterbi(seg, blockd=1)
                            prob_max = hmm_seg.cal_emt_prob(seg, k_max)


                        if prob_max == 0:
                            child_value = inf
                            prob_max = 0.0

                        else:
                            child_value = seg_insert
                            #生成隐藏状态序列
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
    prompt_len = f"Please enter the length for fragmenting DNA sequences (default is 60, current value is {g_line_len}): " if g_lang == "en" else f"请输入DNA序列片段化的长度（默认值为60,当前值为{g_line_len}）:"
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

def standoutput_aliangment(HMM_obj:HMM,sequence,hidden_states):
    #生成参考序列 sequence_ref
    sequence_ref=[]
    for i in range(HMM_obj.dnalen):
        obv_id=np.argmax(HMM_obj.emission_probs[i])
        obv_char=HMM_obj.observations[obv_id]
        sequence_ref.append(obv_char)
    sequence_ref.append(HMM_obj.observations[-1])
    lines_ref= [sequence_ref[i:i+60] for i in range(0, len(sequence_ref), 60)]

    index=0
    operations=[]
    current_ops=hidden_states[index]
    current_ops_pos=get_number_in_string(current_ops)
    pos = 0
    while pos < HMM_obj.dnalen:
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
        line1.append(HMM_obj.observations[-1])
        print(' '*12+"".join(line1))
    return

def generate_model():
    global g_lang
    prompt_len = "Enter the length of the DNA sequence, excluding the terminator (e.g., for ACGe, the length is 3):" if g_lang == "en" else "请输入DNA序列的长度，不含结束符（如 ACGe 长度为3）："
    prompt_h5file = "Enter the filename for the DNA sequence model (e.g., model.hdf5):" if g_lang == "en" else "请输入DNA序列模型文件名（如model.hdf5）:"
    prompt_seqfile = "Enter the filename containing a group of DNA sequences in fasta format:" if g_lang == "en" else "请输入包含一组DNA序列的fasta文件名:"
    dnalen=input(prompt_len)
    dnalen=int(dnalen)
    seqfile=input(prompt_seqfile)
    h5file = input(prompt_h5file)
    hmm = HMM(dnalen)
    hmm.emission_prob_update(seqfile)
    hmm.save_h5(h5file)
    hmm.print_model()

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

    enquire_param()


    for i in range(int(demontimes)):
        obs_seq, hiden_seq, hs_idx_seq = hmm.generate_train_seq()

        #print(obs_seq)
        print_list_characters(obs_seq, 60)
        print("Observation Probability=%e" % (hmm.cal_emt_prob(obs_seq, hs_idx_seq)))
        print(hiden_seq)

        #ln_len = g_line_len
        mytree, myleaf = create_tree(hmm, obs_seq, line_len=g_line_len, max_size_level=g_nodes_per_level, searching_scal=g_sch_scal)
        leaf, pianyi, prob = myleaf[0]
        new_hs = get_hidden_states(leaf, g_line_len)
        new_hs.append(hmm.states[-1])
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
        print(anno)
        #print(obs_seq)
        print_list_characters(obs_seq, 60)

        #ln_len = g_line_len
        mytree, myleaf = create_tree(hmm, obs_seq, line_len=g_line_len, max_size_level=g_nodes_per_level, searching_scal=g_sch_scal)
        leaf, pianyi, prob = myleaf[0]
        new_hs = get_hidden_states(leaf, g_line_len)
        new_hs.append(hmm.states[-1])
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
