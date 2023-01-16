** Generated for: hspiceD
** Generated on: Jan 16 19:00:00 2023
** Design library name: testRohm_n
** Design cell name: INVB_Iris
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
** Cell name: INVB and voltage subtractor
** View name: schematic
.subckt subTractorB out outb v0 vb vbl vdbl
m12 outb net039 net42 net42 P L=180e-9 W=5e-6
m10 out outb net42 net42 P L=180e-9 W=5e-6
m2 opout net013 net42 net42 P L=180e-9 W=4e-6
m1 net013 net37 net42 vdd P L=180e-9 W=400e-9
m0 net42 net37 net37 vdd P L=180e-9 W=400e-9
m11 outb net039 0 0 N L=180e-9 W=2e-6
m9 out outb 0 0 N L=180e-9 W=2e-6
m7 vbl opout blout 0 N L=180e-9 W=50e-6
m8 opout net33 0 0 N L=180e-9 W=4e-6
m6 0 net33 net33 0 N L=180e-9 W=220e-9
m5 net36 net33 0 0 N L=180e-9 W=300e-9
m4 net36 v0 net013 0 N L=180e-9 W=300e-9
m3 net37 vbl net36 0 N L=180e-9 W=300e-9
c1 net013 net45 500e-15
r2 blout 0 50e3
r1 net42 net33 100e3
r0 net45 opout 10e3
v1 net42 0 DC=1.8
m23 net039 blout net038 0 N L=200e-9 W=40e-6
m22 net036 vdbl net038 0 N L=200e-9 W=40e-6
m24 net038 vb 0 0 N L=200e-9 W=40e-6
m20 net039 net036 net42 net42 P L=200e-9 W=40e-6
m21 net42 net036 net036 net42 P L=200e-9 W=40e-6
.ends subTractorB
** End of subcircuit definition.


** Library name: testRohm_n
** Cell name: INVB_iris
** View name: schematic
.subckt INVB_1 invout v0 vbl
m4 net91 v0 net94 0 N L=180e-9 W=300e-9
m3 net92 vbl net91 0 N L=180e-9 W=300e-9
m8 opout net88 0 0 N L=180e-9 W=4e-6
m6 0 net88 net88 0 N L=180e-9 W=220e-9
m5 net91 net88 0 0 N L=180e-9 W=300e-9
m7 vbl opout invout 0 N L=180e-9 W=40e-6
v1 net103 0 DC=1.8
r0 net99 opout 10e3
r1 net103 net88 100e3
r2 invout 0 50e3
c1 net94 net99 500e-15
m2 opout net94 net103 net103 P L=180e-9 W=4e-6
m1 net94 net92 net103 net103 P L=180e-9 W=400e-9
m0 net103 net92 net92 net103 P L=180e-9 W=400e-9
.ends INVB_1
** End of subcircuit definition.


** INVB_1 and INVB_2 differ in the value of r2,
** where as for 8*100*3 NN, #fist layer * r1 =  #second layer * r2, hence

** Library name: testRohm_n
** Cell name: INVB_iris
** View name: schematic
.subckt INVB_2 invout v0 vbl
m4 net91 v0 net94 0 N L=180e-9 W=300e-9
m3 net92 vbl net91 0 N L=180e-9 W=300e-9
m8 opout net88 0 0 N L=180e-9 W=4e-6
m6 0 net88 net88 0 N L=180e-9 W=220e-9
m5 net91 net88 0 0 N L=180e-9 W=300e-9
m7 vbl opout invout 0 N L=180e-9 W=40e-6
v1 net103 0 DC=1.8
r0 net99 opout 10e3
r1 net103 net88 100e3
r2 invout 0 4e3
c1 net94 net99 500e-15
m2 opout net94 net103 net103 P L=180e-9 W=4e-6
m1 net94 net92 net103 net103 P L=180e-9 W=400e-9
m0 net103 net92 net92 net103 P L=180e-9 W=400e-9
.ends INVB_2
** End of subcircuit definition.

.END