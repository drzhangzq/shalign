# shalign
一个基于隐马尔科夫模型（HMM）的DNA序列对齐程序
#程序的功能
程序目前有三个功能模块：1）模型生成：从fasta文件中统计DNA序列每个位置上碱基出现的概率，保存为hdf5格式；2）解码演示：根据模型生成DNA观测序列，并解析观测序列对应的隐藏状态；3）序列解码：从fasta文件中读取DNA观测序例，解析对应的隐藏状态。
程序运行的一般流程：
进入主程序 
    python shalign.py
根据提示选择菜单1）模型生成，根据提示输入DNA序列的长度、用于统计的DNA序列.fasta文件名以及统计后保存HMM模型的.hdf5文件名。HMM模型保存成功后会简要输出模型的参数。
之后可以选择菜单2）解码演示或菜单3）序列解码。
    当选择菜单2后，根据提示输入HMM模型的.hdf5文件名，输入演示的次数，程序会根据HMM模型随机生成DNA观测序列和生成该观测序列对应的隐藏状态序列，之后会根据HMM模型对观测序列进行解码得到隐藏状态序列。可以对比两个隐藏状态序列的区别。
    当选择菜单3后，根据提示输入HMM模型的.hdf5文件名和DNA序列的.fasta文件名，程序会逐条读取fasta文件中的DNA序列，并根据HMM模型对DNA序列进行解码，输出解码后的隐藏状态序列。

# Shalign
This is a DNA sequence alignment program based on Hidden Markov Models (HMM).
#Program Features 
The program currently consists of three functional modules: 1) Model Generation: It calculates the probability of each base appearing at each position in the DNA sequence from a fasta file and saves it in hdf5 format. 2) Decoding Demonstration: It generates DNA observation sequences based on the model and decodes the corresponding hidden states of the observation sequences. 3) Sequence Decoding: It reads DNA observation sequences from a fasta file and decodes the corresponding hidden states.
The general workflow of the program is as follows: 
    Enter the main program using 'python shalign.py.' 
    Follow the prompts to choose Menu 1) Model Generation, where you input the length of the DNA sequence, the filename of the DNA sequence used for statistics (.fasta), and the filename for saving the HMM model after statistics (.hdf5). After successful saving of the HMM model, the program briefly outputs the model parameters. 
    Subsequently, you can choose Menu 2) Decoding Demonstration or Menu 3) Sequence Decoding. 
    When selecting Menu 2, follow the prompts to input the filename of the saved HMM model (.hdf5) and the number of demonstrations. The program will randomly generate DNA observation sequences based on the HMM model and generate the corresponding hidden state sequences. It then decodes the observation sequences using the HMM model to obtain the hidden state sequences, allowing for a comparison of the differences between two hidden state sequences. 
    When choosing Menu 3, follow the prompts to input the filename of the saved HMM model (.hdf5) and the filename of the DNA sequence (.fasta). The program will read DNA sequences from the fasta file one by one and decode the DNA sequences based on the HMM model, outputting the decoded hidden state sequences.
