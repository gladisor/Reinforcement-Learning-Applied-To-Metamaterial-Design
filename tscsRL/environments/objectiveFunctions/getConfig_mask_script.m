clear all; clc;

M = 4;
d_min = 2.1

config = [1.5, -2.4, 1.2, 2.5, 2.1, 1.9, -1.2, -3.4];
invalid_cyl = [2, 3]

config_mask = getConfig_Mask(config, invalid_cyl, M, d_min)