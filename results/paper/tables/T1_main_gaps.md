run                step  measure            n    Δppl            95% CI      win
lora-only-matched  200   training         106  +1.610  [+1.293, +2.013]  *  1.00
hebbian            200   training         106  +1.653  [+1.335, +2.057]  *  1.00
delta_chunk        200   training         106  +1.670  [+1.349, +2.074]  *  1.00
lora-only-matched  400   training         106  +1.784  [+1.440, +2.222]  *  1.00
hebbian            400   training         106  +1.852  [+1.499, +2.300]  *  1.00
delta_chunk        400   training         106  +1.909  [+1.542, +2.364]  *  1.00
lora-only-matched  600   training         106  +1.859  [+1.503, +2.310]  *  1.00
hebbian            600   training         106  +1.928  [+1.561, +2.392]  *  1.00
delta_chunk        600   training         106  +1.992  [+1.613, +2.465]  *  1.00
lora-only-matched  200   within           106  +0.000  [+0.000, +0.000]     0.50
hebbian            200   within           106  +0.016  [+0.005, +0.028]  *  0.72
delta_chunk        200   within           106  +0.144  [+0.117, +0.179]  *  1.00
lora-only-matched  400   within           106  +0.000  [+0.000, +0.000]     0.50
hebbian            400   within           106  +0.014  [+0.009, +0.020]  *  0.79
delta_chunk        400   within           106  +0.059  [+0.047, +0.073]  *  0.99
lora-only-matched  600   within           106  +0.000  [+0.000, +0.000]     0.50
hebbian            600   within           106  +0.034  [+0.026, +0.045]  *  0.89
delta_chunk        600   within           106  +0.058  [+0.046, +0.073]  *  0.96
lora-only-matched  200   between_slices   106  +0.000  [+0.000, +0.000]     0.50
hebbian            200   between_slices   106  +0.014  [+0.008, +0.021]  *  0.70
delta_chunk        200   between_slices   106  +0.027  [+0.019, +0.037]  *  0.83
lora-only-matched  400   between_slices   106  +0.000  [+0.000, +0.000]     0.50
hebbian            400   between_slices   106  -0.002  [-0.005, +0.001]     0.46
delta_chunk        400   between_slices   106  +0.007  [+0.003, +0.012]  *  0.65
lora-only-matched  600   between_slices   106  +0.000  [+0.000, +0.000]     0.50
hebbian            600   between_slices   106  +0.002  [-0.000, +0.006]     0.54
delta_chunk        600   between_slices   106  +0.012  [+0.008, +0.018]  *  0.74
lora-only-matched  200   total            106  +1.610  [+1.303, +2.003]  *  1.00
hebbian            200   total            106  +1.683  [+1.361, +2.092]  *  1.00
delta_chunk        200   total            106  +1.842  [+1.489, +2.284]  *  1.00
lora-only-matched  400   total            106  +1.784  [+1.450, +2.213]  *  1.00
hebbian            400   total            106  +1.864  [+1.517, +2.306]  *  1.00
delta_chunk        400   total            106  +1.975  [+1.601, +2.442]  *  1.00
lora-only-matched  600   total            106  +1.859  [+1.513, +2.301]  *  1.00
hebbian            600   total            106  +1.965  [+1.596, +2.432]  *  1.00
delta_chunk        600   total            106  +2.062  [+1.675, +2.544]  *  1.00
hebbian            200   vs_matched_lora  106  +0.073  [+0.057, +0.094]  *  0.91
delta_chunk        200   vs_matched_lora  106  +0.232  [+0.190, +0.280]  *  1.00
hebbian            400   vs_matched_lora  106  +0.080  [+0.066, +0.098]  *  0.99
delta_chunk        400   vs_matched_lora  106  +0.191  [+0.156, +0.231]  *  0.99
hebbian            600   vs_matched_lora  106  +0.106  [+0.086, +0.130]  *  1.00
delta_chunk        600   vs_matched_lora  106  +0.203  [+0.165, +0.249]  *  1.00
hebbian            200   cross_doc         24  +0.008  [-0.004, +0.024]     0.50
delta_chunk        200   cross_doc         24  +0.003  [-0.006, +0.012]     0.58
hebbian            400   cross_doc         24  -0.003  [-0.010, +0.002]     0.50
delta_chunk        400   cross_doc         24  +0.006  [-0.000, +0.013]     0.67
hebbian            600   cross_doc         24  -0.004  [-0.015, +0.006]     0.38
delta_chunk        600   cross_doc         24  -0.002  [-0.006, +0.002]     0.38
hebbian            200   cross_slice       30  +0.014  [+0.003, +0.030]  *  0.83
delta_chunk        200   cross_slice       30  +0.027  [+0.013, +0.048]  *  0.77
hebbian            400   cross_slice       30  +0.004  [-0.002, +0.011]     0.60
delta_chunk        400   cross_slice       30  +0.002  [-0.004, +0.008]     0.60
hebbian            600   cross_slice       30  +0.005  [-0.001, +0.014]     0.60
delta_chunk        600   cross_slice       30  +0.015  [+0.005, +0.029]  *  0.73
