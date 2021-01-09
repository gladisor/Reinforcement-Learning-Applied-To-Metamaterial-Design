function Q = getMetric_thinShells_radii_material(x,M,av,c_pv,rho_shv, k0amax, k0amin, nfreq)
% if max(size(gcp)) == 0 % parallel pool needed
%     parpool % create the parallel pool
% end

hav = av/10; %thicknesses of  thin shells
% %%%%%%%%%%%%%%%%% Properties of water  %%%%%%%%%%%%%%%%%%%%%%%%%%
rho =1000;  c0=1480;  
xM=(reshape(x',2,M))'; % position vectors
XM= xM(:,1);
YM= xM(:,2);
absr= sqrt(XM.^2 + YM.^2);          % |r_j|
argr = atan2(YM,XM);                % argument of vector r_j
if argr <0 
    argr= argr+2*pi;
end
%%
% freqmax = (k0amax)*c0/(2*pi*a);        %max freq to give ka=20
% freqmin = (k0amin)*c0/(2*pi*a); 
freqmax = (k0amax)*c0/(2*pi*max(av));        %max freq to give ka=20
freqmin = (k0amin)*c0/(2*pi*max(av)); 
df=(freqmax-freqmin)/(nfreq-1);
freqv = freqmin:df:freqmax; 
% kav = zeros(size(freqv));
kav = zeros(nfreq,M);
kav_max=zeros(nfreq,1);
Q =zeros(nfreq,1);

for Ifreq=1:length(freqv)
    freq=freqv(Ifreq);   omega = 2*pi*freq;    
    k0 = omega/c0;
    ka=k0*av;
    %kav(Ifreq)=ka;
    
%%%% give n, get matrix, solve it
%     nmax = round(2.5*ka);
    nmax = round(2.5*max(ka));
    N=nmax;
    nv = -nmax:nmax;    
% %%% T matrix for thin elastic shells
T_1 =zeros((2*N+1),(2*N+1),M);
for j = 1:M
  T_1(:,:,j) = T_shell(ka(j),nv,rho,c0,hav(j),c_pv(j),rho_shv(j));
end

Ainv=zeros((2*N+1),M);
AM=zeros((2*N+1),M);
% Jpvka=zeros((2*N+1),M);
% Hpvka=zeros((2*N+1),M);
% % % % % T matrix  for  Rigid cilinders
% for j = 1:M
%     Jpvka(:,j) =(besselj(nv'-1,ka(j)) -  besselj(nv'+1,ka(j)))/2;
%     Hpvka(:,j) =(besselh(nv'-1,ka(j)) -  besselh(nv'+1,ka(j)))/2;
%     T_1(:,:,j)= diag( -( Jpvka(:,j) )./ ( Hpvka(:,j) ) );
% %     T{j} = T(:,j);
% end

T = cell(1,M);
for j=1:M
    T{j} = T_1(:,:,j);
end
Tdiag = blkdiag(T{:});
%%
for j=1:M
    Ainv(:,j)=exp(1i*k0*XM(j))* exp(1i*nv'*pi/2  );      %%% Incident Plain wave amplitude 
    AM(:,j) = T_1(:,:,j) * Ainv(:,j);
end
verAv = reshape(AM, M*(2*N+1) ,1); % verCV is a colunm vector of length (2*N+1)*RN*M
%%
Xbig = zeros( M*(2*N+1) );    absrjm = zeros(M);    argrjm = zeros(M);  
P__jm = zeros(2*N+1); 

if M==1
   Xbig = eye(2*N+1);
else
for j = 1:M %loop over cylinder j with cylinder m
%     j;
   for m = 1:M
%    m;
       if m==j
            Xbig( ((j-1)*(2*N+1)+1:j*(2*N+1)), ((m-1)*(2*N+1)+1:m*(2*N+1)) ) = eye(2*N+1);
        else
            rjm = xM(j,:) - xM(m,:);   % relative position
            xjm=rjm(1);    
            yjm=rjm(2);
            absrjm(j,m) = norm(rjm);    %  norm of rjm vector 
            argrjm(j,m) = atan2(rjm(2),rjm(1)); %  argument of rjm vector
            if argrjm(j,m) <0
                argrjm(j,m)= argrjm(j,m)+2*pi;
            end

            for I=1:numel(nv)
%                 I;
                n=nv(I);
                Hvrjm =  besselh(n-nv',k0*absrjm(j,m));
                exprjm = exp( 1i*(n-nv')*argrjm(j,m) );
                P__jm(:,I) = Hvrjm  .* exprjm;   
            end              
            
            Xbig( ((j-1)*(2*N+1)+1:j*(2*N+1)),((m-1)*(2*N+1)+1:m*(2*N+1)) ) = - Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) * P__jm;    
       end % ends if statement over cylinder      
   end
end
end
%%
Bv = Xbig \  verAv;         

C_nm=zeros((2*N+1),M);
    for I=1:numel(nv)
        n= nv(I);
        for m=1:M
        C_nm(I,m)=(-1i)^n *  exp(-1i*(k0)*absr(m)*cos(argr(m)));
        end
    end
Cv=reshape(C_nm,M*(2*N+1),1);

Q(Ifreq) = - 4/k0* real((Cv.') * Bv); 
kav_max(Ifreq)=max(ka);
% kav(Ifreq,:)=ka;
end
% 
% plot(kav_max,Q)
% grid on
end
