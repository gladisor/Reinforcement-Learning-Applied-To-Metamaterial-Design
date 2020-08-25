function Q_RMS = getTSCS4CYL(x1, y1, x2, y2, x3, y3, x4, y4)

a = 1;
aa = 1;
M = 4;
ha = aa/10;
c_p = 5480;
rho_sh = 8850;
k0amax = 0.5;
k0amin = 0.3;
nfreq = 11;

x = [x1, y1, x2, y2, x3, y3, x4, y4].';
[Q_RMS,qV,kav,Q] = objectiveFunctionTSCS_RMSka_min_max(x,a,aa,M,ha,c_p,rho_sh,k0amax,k0amin,nfreq);