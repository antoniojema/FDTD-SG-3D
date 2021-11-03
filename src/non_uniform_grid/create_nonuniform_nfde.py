
file_str = \
"""********************* SOLVER OPTIONS ***********************

!SIMULATION
-cfl 0.8

************************* GENERAL **************************

!TIMESTEPS
2.309e-11 1999

*********************** SPACE STEPS ************************
* create by confMesher
!NEWSPACESTEPS
!!XCONS
    0.0
    0 10
    0.1
!!YCONS
    0.0
    0 10
    0.1
!!ZVARI
    0.0
    0 1040
"""

for i in range(500):
    file_str += "    0.1\n"
for i in range(40):
    file_str += "    0.05\n"
for i in range(500):
    file_str += "    0.1\n"

file_str += """
************************ BACKGROUND ************************

!BACKGROUND
!!LINEAR
0 8.85419e-12 1.25664e-06 0

******************* BOUNDARY CONDITIONS ********************

!BOUNDARY CONDITION
PEC XL
PEC XU
PMC YL
PMC YU
PML ZL
10 2.0 0.001
PML ZU
10 2.0 0.001

********************* EXTENDED SOURCES *********************

!PLANE WAVE SOURCE
gauss.gen
locked
-10 -10 495
 20  20 1100
0.0000	0.0000	1.5708	0.0000

******************* VOLUMIC CURRENT PROBES *****************

*p1_log
!NEW PROBE
!!NUMER
!!!FREQ
1e+03 2e+09 1e+05
EX 5 6 490
EY 5 6 490
EZ 5 6 490

!END
"""

with open("non_uniform.nfde", "w") as fout:
    fout.write(file_str)
