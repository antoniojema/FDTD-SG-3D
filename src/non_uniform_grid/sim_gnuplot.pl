set term x11 persist           1
plot 'sim_p1_log_Ex_5_6_490.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'sim_p1_log_Ey_5_6_490.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'sim_p1_log_Ez_5_6_490.dat' using 1:2 every 1::2 with lines
