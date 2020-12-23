function [Q, qV] = getMetric(x, M, k0amax, k0amin, nfreq)

a = 1;
aa = 1;
ha = aa/10;
c_p = 5480;
rho_sh = 8850;
if max(size(gcp)) == 0 % parallel pool needed
	parpool % create the parallel pool
end

[Q_RMS,qV,kav,Q] = objectiveFunctionTSCS_RMSka_min_max(x,a,aa,M,ha,c_p,rho_sh,k0amax,k0amin,nfreq);