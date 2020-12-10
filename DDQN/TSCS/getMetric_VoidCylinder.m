function Q = getMetric_VoidCylinder(x, M, k0amax, k0amin, nfreq)

if max(size(gcp)) == 0 % parallel pool needed
	parpool % create the parallel pool
end

a = 1;

% % %%%%%%%%%%%%%%%%% Properties of nickel thin shell  %%%%%%%%%%%%%%%%%%%%%%%%%%
% ha = a/10; %thickness of  thin shell 
% c_p = 5480;
% rho_sh = 8850;

% %%%%%%%%%%%%%%%%% Properties of water  %%%%%%%%%%%%%%%%%%%%%%%%%%
rho =1000;  c0=1480;  
%%%%%%%%%% position vectors %%%%%%%%%%%%%%%%%
xM=(reshape(x',2,M))';
XM= xM(:,1);
YM= xM(:,2);
absr= sqrt(XM.^2 + YM.^2);          % |r_j|
argr = atan2(YM,XM);                % argument of vector r_j
if argr <0 
    argr= argr+2*pi;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
freqmax = (k0amax)*c0/(2*pi*a);        %max freq to give ka=20
freqmin = (k0amin)*c0/(2*pi*a); 
df=(freqmax-freqmin)/(nfreq-1);
freqv = freqmin:df:freqmax; 
kav = zeros(size(freqv));
Q =zeros(nfreq,1);

parfor Ifreq=1:length(freqv)
    freq=freqv(Ifreq);   omega = 2*pi*freq;    
    k0 = omega/c0;
    ka=k0*a;
    kav(Ifreq)=ka;
%%%% give n, get matrix, solve it
    nmax = round(2.5*ka);
    N=nmax;
    nv = -nmax:nmax;
 % % % % T matrix  for  Void cilinders
    Jvka =besselj(nv',ka);
    Hvka =besselh(nv',ka);
    T_1 = diag( -( Jvka )./ ( Hvka ) );
% % % % % T matrix  for  Rigid cilinders
%     Jpvka =(besselj(nv'-1,ka) -  besselj(nv'+1,ka))/2;
%     Hpvka =(besselh(nv'-1,ka) -  besselh(nv'+1,ka))/2;
%     T_1 = diag( -( Jpvka )./ ( Hpvka ) );
% %%% T matrix for thin elastic shells
%     T_1 = T_shell(ka,nv,rho,c0,ha,c_p,rho_sh);
    T = cell(1,M);
    for j=1:M
        T{j} = T_1;
    end
    Tdiag = blkdiag(T{:}); 
%%
    Ainv=zeros((2*N+1),M);
    AM=zeros((2*N+1),M);
    for j=1:M
        Ainv(:,j)=exp(1i*k0*XM(j))* exp(1i*nv'*pi/2  );      %%% Incident Plain wave amplitude 
        AM(:,j) = T_1 * Ainv(:,j);
    end
    verAv = reshape(AM, M*(2*N+1) ,1); % verCV is a colunm vector of length (2*N+1)*RN*M
%%
Xbig = zeros( M*(2*N+1) );    absrjm = zeros(M);    argrjm = zeros(M);  
P__jm = zeros(2*N+1); 

if M==1
   Xbig = eye(2*N+1);
else
for j = 1:M %loop over cylinder j with cylinder m
    j;
   for m = 1:M
   m;
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
                I;
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

Q(Ifreq) = - 4/k0* real((Cv.') * Bv) ;
end
end
