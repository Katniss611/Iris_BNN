** Generated for: hspiceD
** Generated on: Jan 15 22:05:11 2023
** Design library name: testRohm_n
** Design cell name: BNN_IRIS_A_1
** Design view name: schematic


.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2

** Library name: testRohm_n
** Cell name: INVA_iris_1
** View name: schematic
.subckt INVA_iris_1 out outb vb vdbl vin
m0 outb net77 0 0 N L=180e-9 W=2e-6
m2 out outb 0 0 N L=180e-9 W=2e-6
m1 outb net77 net78 net78 P L=180e-9 W=5e-6
m3 out outb net78 net78 P L=180e-9 W=5e-6
m23 net77 vdbl net76 0 N L=200e-9 W=40e-6
m22 net75 vin net76 0 N L=200e-9 W=40e-6
m24 net76 vb 0 0 N L=200e-9 W=40e-6
r0 vin 0 112e3
m20 net77 net75 net78 net78 P L=200e-9 W=40e-6
m21 net78 net75 net75 net78 P L=200e-9 W=40e-6
v0 net78 0 DC=1.8
.ends INVA_iris_1
** End of subcircuit definition.

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