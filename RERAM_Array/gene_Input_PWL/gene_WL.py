# -*- coding: utf-8 -*-
# @Author  : Tian Hongrong
# @Time    : 2023/1/14 20:11
# @Function: create stimulus file for Iris dataset with binary neural network(BNN) based circuit

import numpy as np
output_file = open("iris.inp", "w")

"""
    transient analysis and parameters setting
"""

output_file.write(".PARAM mvdd=1.8\n")
output_file.write(".TRAN 1u 121u\n")
output_file.write("VVDD VDD 0 DC mvdd\n")
output_file.write("VG G 0 DC 0\n")
# for type-C I-V converter
output_file.write("VV0 V0 0 DC 0.9\n")
output_file.write("VVB VB 0 DC 0.6\n")
output_file.write("\n")
output_file.write(".PROBE V(*) I(*) \n")
output_file.write(".PROBE tran v(OUT1) \n")
output_file.write(".PROBE tran v(OUT2) \n")
output_file.write(".PROBE tran v(OUT3) \n")
output_file.write("\n")
output_file.write(".include \"../bu40n1.mdl\"\n")
output_file.write(".lib \"../bu40n1.skw\" NT\n")
output_file.write(".lib \"../bu40n1.skw\" PT\n")
output_file.write("\n")


"""
    the input(stimulus) of 150 Iris flower data are divided into 8 groups(since I use 2-bit to represent
different data in software training phase),so the WL and WLb are designed into 8 pairs, and initial
pre-charge state is mvdd(high volt)
"""


new_bi_data = np.load('binary_input_data.npy', allow_pickle=True)


# def sign(x):
#     if x > 0:
#         return 1.8
#     elif x <= 0:
#         return 0


WL = np.zeros(shape=new_bi_data.shape)
WLB = np.zeros(shape=new_bi_data.shape)

for i in range(new_bi_data.shape[0]):
    for j in range(new_bi_data.shape[1]):
        WL[i][j] = new_bi_data[i][j]
        # WL[i][j] = sign(new_bi_data[i][j])

for i in range(new_bi_data.shape[0]):
    for j in range(new_bi_data.shape[1]):
        WLB[i][j] = (-new_bi_data[i][j])
        # WLB[i][j] = sign(-new_bi_data[i][j])

for i in range(1, 9):
    output_file.write("VWL1_%d WL1_%d 0 PWL \n" % (i, i))
    for j in range(1, 151):
        if j == 1:
            output_file.write("+    0n  %s \n" % "mvdd")
        else:
            output_file.write("+    %dn %s \n" % ((800 * (j - 1) + 1), "mvdd"))

        output_file.write("+    %dn %s \n" % ((800 * (j - 1) + 300), "mvdd"))
        # use WL[j][i] since input matrix is[150,8]
        output_file.write("+    %dn %s \n" % ((800 * (j - 1) + 300 + 1), "mvdd" if WL[j-1][i-1] == 1 else 0))
        output_file.write("+    %dn %s \n" % ((800 * j), "mvdd" if WL[j-1][i-1] == 1 else 0))


for i in range(1, 9):
        output_file.write("VWL1_%db WL1_%db 0 PWL \n" % (i, i))
        for j in range(1, 151):
            if j == 1:
                output_file.write("+    0n  %s \n" % "mvdd")
            else:
                output_file.write("+    %dn %s \n" % ((800 * (j - 1) + 1), "mvdd"))

            output_file.write("+    %dn %s \n" % ((800 * (j - 1) + 300), "mvdd"))
            output_file.write("+    %dn %s \n" % ((800 * (j - 1) + 300 + 1), "mvdd" if WLB[j-1][i-1] == 1 else 0))
            output_file.write("+    %dn %s \n" % ((800 * j), "mvdd" if WLB[j-1][i-1] == 1 else 0))


"""
    add .meas command to monitor signal
"""

# output_file.write("\n")
# for i in range(1, 151):
#     output_file.write(".measure tran avgOut1_%d AVG v(OUT1) FROM = %du TO =%du\n" % (i, 800*i-1, 800*i))
#     output_file.write(".measure tran avgOut2_%d AVG v(OUT2) FROM = %du TO =%du\n" % (i, 800 * i - 1, 800 * i))
#     output_file.write(".measure tran avgOut3_%d AVG v(OUT3) FROM = %du TO =%du\n" % (i, 800 * i - 1, 800 * i))

output_file.close()

