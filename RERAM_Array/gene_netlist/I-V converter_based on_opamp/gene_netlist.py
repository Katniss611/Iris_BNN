# -*- coding: utf-8 -*-
# @Author  : Tian Hongrong
# @Time    : 2023/1/13 20:20
# @Function: Generate netlist using I-V converter based on operational amplifier
#            by including module_header_C.sp

import numpy as np

middle_weight = np.load('../../optimizedWeight/2bit_90.6%/middle_weight_100hid_2bit.npy', allow_pickle=True, encoding="latin1")
middle_weight = np.sign(middle_weight)
output_weight = np.load('../../optimizedWeight/2bit_90.6%/output_weight_100hid_2bit.npy', allow_pickle=True, encoding="latin1")
output_weight = np.sign(output_weight)

output_file = open("netlist", "w")
header_file = open("../../module_header_C.sp", "r")
line = header_file.readline()
while (not line == ""):
    output_file.write(line)
    line = header_file.readline()

output_file.write("\n\n")
# ==================layer1===================
# .subckt cellP vbl wl11 wl11b
# .subckt cellN vbl wl11 wl11b

m = len(middle_weight[0])
n = len(middle_weight[1])
for i in range(0, 8):
    for j in range(0, 100):
        # eg. xl1i1 bl1_1 wl1_1 wl1_1b cellp
        output_file.write("xl1i%d bl1_%d wl1_%d wl1_%db %s\n" % ((100*i+(j+1)), (j+1), (i+1), (i+1),
                                                               "cellP" if middle_weight[i][j] == 1 else "cellN"))
# .subckt INVC_iris out outb v0 vb vbl vdbl
# serve as a voltage subtractor and I-V converter C, combined in INVC_iris

# first layer's voltage subtractor,dummy bitline on positive "+" terminal
for j in range(0, 100):
    output_file.write("xl1Subtractor%d wl2_%d wl2_%db v0 vb bl1_%d netvref INVC_iris \n" %((j+1), (j+1), (j+1), (j+1)))


# ==================layer2===================

for i in range(0, 100):
    for j in range(0, 3):
#         eg. xl2i2 bl2_1 wl2_1 wl2_1b cellp
        output_file.write("xl2i%d bl2_%d wl2_%d wl2_%db %s\n" % ((3*i+(j+1)), (j+1), (i+1), (i+1),
                                                               "cellP" if output_weight[i][j] == 1 else "cellN"))
# second layer's I-V converter
# .subckt INVC_2 out v0 vb vbl
for i in range(0, 3):
    output_file.write("xl2INVi%d out%d v0 vb bl2_%d INVC_2\n" % ((i+1), (i+1), (i+1)))



# ==================dummy BL===================
for i in range(0, 8):
    # net137 is the BL of dummy BL
    output_file.write("xdbli%d net137 vdd 0 %s\n" %((i + 1), "cellP" if i % 2 == 0 else "cellN"))

# ==================dummy BL's I-V converter===================
# .subckt INVC_1 out v0 vb vbl
output_file.write("xrefi netvref v0 vb net137 INVC_1\n")

output_file.write("\n")
output_file.write(".END\n")
output_file.close()