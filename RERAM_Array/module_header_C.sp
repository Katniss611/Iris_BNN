** Generated for: hspiceD
** Generated on: Jan 13 14:05:32 2023
** Design library name: testRohm_n
** Design cell name: BNN_IRIS_C
** Design view name: schematic


.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2

** Library name: testRohm_n
** Cell name: cellP
** View name: schematic
.subckt cellP vbl wl11 wl11b
v0 net30 0 DC=1.8
m0 net30 wl11 net28 net30 P L=180e-9 W=5e-6
m1 net29 wl11b net30 net30 P L=180e-9 W=5e-6
r0 net28 vbl R 10e6
r1 net29 vbl R 900e3
.ends cellP
** End of subcircuit definition.

** Library name: testRohm_n
** Cell name: cellN
** View name: schematic
.subckt cellN vbl wl11 wl11b
v0 net30 0 DC=1.8
m0 net30 wl11 net28 net30 P L=180e-9 W=5e-6
m1 net29 wl11b net30 net30 P L=180e-9 W=5e-6
r0 net28 vbl R 900e3
r1 net29 vbl R 10e6
.ends cellN
** End of subcircuit definition.

** Library name: testRohm_n
** Cell name: INVC_1
** View name: schematic
.subckt INVC_1 out v0 vb vbl
rl vbl out 112e3
m6 net36 v0 net37 0 N L=200e-9 W=40e-6
m5 out vbl net37 0 N L=200e-9 W=40e-6
m4 net37 vb 0 0 N L=200e-9 W=40e-6
m8 net42 net36 net36 net42 P L=200e-9 W=40e-6
m7 out net36 net42 net42 P L=200e-9 W=40e-6
vvdc net42 0 DC=1.8
.ends INVC_1
** End of subcircuit definition.

** Library name: testRohm_n
** Cell name: INVC_iris
** View name: schematic
.subckt INVC_iris out outb v0 vb vbl vdbl
m0 outb net139 0 0 N L=180e-9 W=2e-6
m2 out outb 0 0 N L=180e-9 W=2e-6
m1 outb net139 net134 net134 P L=180e-9 W=5e-6
m3 out outb net134 net134 P L=180e-9 W=5e-6
m6 net131 v0 net132 0 N L=200e-9 W=40e-6
m5 net133 vbl net132 0 N L=200e-9 W=40e-6
m4 net132 vb 0 0 N L=200e-9 W=40e-6
m23 net139 vdbl net138 0 N L=200e-9 W=40e-6
m22 net137 net133 net138 0 N L=200e-9 W=40e-6
m24 net138 vb 0 0 N L=200e-9 W=40e-6
rl vbl net133 112e3
m8 net134 net131 net131 net134 P L=200e-9 W=40e-6
m7 net133 net131 net134 net134 P L=200e-9 W=40e-6
m20 net139 net137 net134 net134 P L=200e-9 W=40e-6
m21 net134 net137 net137 net134 P L=200e-9 W=40e-6
vvdc net134 0 DC=1.8
.ends INVC_iris
** End of subcircuit definition.

** Library name: testRohm_n
** Cell name: INVC_2
** View name: schematic
.subckt INVC_2 out v0 vb vbl
vvdc net61 0 DC=1.8
rl vbl out 9e3
m6 net59 v0 net60 0 N L=200e-9 W=40e-6
m5 out vbl net60 0 N L=200e-9 W=40e-6
m4 net60 vb 0 0 N L=200e-9 W=40e-6
m8 net61 net59 net59 net61 P L=200e-9 W=40e-6
m7 out net59 net61 net61 P L=200e-9 W=40e-6
.ends INVC_2
** End of subcircuit definition.