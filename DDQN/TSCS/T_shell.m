%%% This code determines the T-matrix for a submerged elastic thin shell
function [T_0] = T_shell(ka,nv,rho,c,ha,c_p,rho_sh)
        % T_0 - empty shell    
%%%%%%%%%%%%%%%%%   Shell properties   %%%%%%%%%%%%%%%%%%%%%%%%%
% ha = 0.025; c_p = 6420; rho_sh = 2700;
%%%%%%%%%%%%%%%%% Properties of water  %%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 1e3;  c = 1.48e3; %c = 1.475e3;
N = (length(nv)-1)/2;
%   Shell thickness
beta = ha;%ha/sqrt(12);
%   External fluid, water
K = rho*(c^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Dimensionless structural frequency
Om = (c/c_p).*ka;
%   Bessel functions and their derivatives
J_n = besselj(nv,ka);
Jp_n = 0.5*( besselj(nv-1,ka) - besselj(nv+1,ka));
H_n = besselh(nv,ka);
Hp_n = 0.5*( besselh(nv-1,ka) - besselh(nv+1,ka));
%   Impedances
D_n = ((Om.^4) - (Om.^2).*(1+(nv.^2)+((beta.^2).*(nv.^4)))+ (beta^2).*(nv.^6) )./(Om.*((Om.^2)-(nv.^2)));
Zs_n = -1i*rho_sh*c_p*ha.*D_n; %shell
Z_n = 1i*rho*c.*H_n./Hp_n; %acoustic, outgoing
Zb_n = 1i*rho*c.*J_n./Jp_n; %acoustic, incoming
T_n  = -(Zs_n+Zb_n)./(Zs_n+Z_n).*(Jp_n./Hp_n); 
%   Empty shell T-matrix
T_0 = diag(T_n);
end