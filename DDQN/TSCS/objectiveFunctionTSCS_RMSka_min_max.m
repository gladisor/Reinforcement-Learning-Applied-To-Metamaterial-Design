function [Q_RMS,qV,kav,Q] = objectiveFunctionTSCS_RMSka_min_max(x,a,aa,M,ha,c_p,rho_sh,k0amax,k0amin,nfreq)
xM=(reshape(x',2,M))';
XM= [xM(:,1)];
YM= [xM(:,2)];
absr= sqrt(XM.^2 + YM.^2);          % |r_j|
argr = atan2(YM,XM);                % argument of vector r_j
if argr <0 
    argr= argr+2*pi;
end
% %%%%%%%%%%%%%%%%% Properties of water  %%%%%%%%%%%%%%%%%%%%%%%%%%
rho =1000;  c0=1480;  %kap0=rho*(c0)^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
freqmax = (k0amax)*c0/(2*pi*a);        %max freq to give ka=20
freqmin = (k0amin)*c0/(2*pi*a); 
df=(freqmax-freqmin)/(nfreq-1);
freqv = freqmin:df:freqmax; 
kav = zeros(size(freqv));
Q =zeros(nfreq,1);
s_j = zeros(nfreq,M,2) ;
q_j = zeros(nfreq,M,2) ;

for Ifreq=1:length(freqv)
    freq=freqv(Ifreq);   omega = 2*pi*freq;    
    k0 = omega/c0;
    ka=k0*a;
    kaa=k0*aa;
    kav(Ifreq)=ka;
%%%% give n, get matrix, solve it
 nmax = round(2.5*ka);
 N=nmax;
 nv = -nmax:nmax;
% %  THE T_n diagonal matrix .....  
% % T matrix  for inner Rigid cilinders
Jpvka =(besselj(nv'-1,ka) -  besselj(nv'+1,ka))/2;
Hpvka =(besselh(nv'-1,ka) -  besselh(nv'+1,ka))/2;
T_0 = diag( -( Jpvka )./ ( Hpvka ) );
% % % T matrix  for outer cloaking cilinders
Jpvkaa =(besselj(nv'-1,kaa) -  besselj(nv'+1,kaa))/2;
Hpvkaa =(besselh(nv'-1,kaa) -  besselh(nv'+1,kaa))/2;
T_1 = diag( -( Jpvkaa )./ ( Hpvkaa ) );
%%% T matrix for thin elastic shells
% T_0 = T_shell(ka,nv,rho,c0,ha,c_p,rho_sh);
% T_1=T_0;
T = cell(1,M);
T{1}=T_0;
    for j=2:M
        T{j} = T_1;
    end
Tdiag = blkdiag(T{:}); 
%%
%  MAKING THE RIGHT HAND SIDE: Cv= [T_n]*[A_n] VECTOR .....    
% s = '	MAKING THE RIGHT HAND SIDE:  Cv= [T_n]*[A_n] VECTOR .....'
Ainv=zeros((2*N+1),M);
AM=zeros((2*N+1),M);
Ainv(:,1)=exp(1i*k0*XM(1))* exp(1i*nv'*pi/2  );      %%% Incident Plain wave amplitude 
AM(:,1) = T_0 * Ainv(:,1);
    for j=2:M
        Ainv(:,j)=exp(1i*k0*XM(j))* exp(1i*nv'*pi/2  );      %%% Incident Plain wave amplitude 
        AM(:,j) = T_1 * Ainv(:,j);
    end
verAv = reshape(AM, M*(2*N+1) ,1); % verCV is a colunm vector of length (2*N+1)*RN*M
%%
% %	MAKING THE BIGMATRICES:   Xbig and gradXbig 
%  s = '	MAKING THE BIGMATRICES:   Xbig and gradXbig '

Xbig = zeros( M*(2*N+1) );    absrjm = zeros(M);    argrjm = zeros(M);  
P__jm = zeros(2*N+1); 
gradP_jm=zeros(2*N+1,2*N+1,M,M,2); 
gradP_mj=zeros(2*N+1,2*N+1,M,M,2);
gradXbigM=zeros(2*N+1,2*N+1,M,M,2);
gradXbigM1=zeros(2*N+1,2*N+1,M,M,2);
gradXbig=zeros( M*(2*N+1),M*(2*N+1),M,2 );
Dp_jm =  zeros(2*N+1); Dh_jm =  zeros(2*N+1);
Dp_mj =  zeros(2*N+1); Dh_mj =  zeros(2*N+1);
if M==1
   Xbig = eye(2*N+1);
   gradXbig=zeros(2*N+1,2*N+1,2);
else
for j = 1:M; %loop over cylinder j with cylinder m
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
                Hpvrjm =  ( besselh(n-nv'-1,k0*absrjm(j,m)) - besselh(n-nv'+1,k0*absrjm(j,m)) )/2;
                exprjm = exp( 1i*(n-nv')*argrjm(j,m) );
                Dp_jm(:,I) = Hpvrjm .* exprjm;
                P__jm(:,I) = Hvrjm  .* exprjm;
                Dh_jm(:,I)= 1i*(n-nv').* Hvrjm .* exprjm;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                exprmj = exp( 1i*(n-nv')*(argrjm(j,m)+pi) );
                Dp_mj(:,I) = Hpvrjm .* exprmj;
                Dh_mj(:,I)= 1i*(n-nv').* Hvrjm .* exprmj    ;      
            end              
            
            Xbig( ((j-1)*(2*N+1)+1:j*(2*N+1)),((m-1)*(2*N+1)+1:m*(2*N+1)) ) = - Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) * P__jm;    
            gradP_jm(:,:,m,j,1)= k0*xjm*Dp_jm/absrjm(j,m) - (Dh_jm*yjm/((absrjm(j,m))^2));
            gradP_jm(:,:,m,j,2)= k0*yjm*Dp_jm/absrjm(j,m) + (Dh_jm*xjm/((absrjm(j,m))^2));
            gradXbigM(:,:,m,j,1) = - Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) *gradP_jm(:,:,m,j,1);
            gradXbigM(:,:,m,j,2) = - Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) *gradP_jm(:,:,m,j,2)   ;

            gradXbig(((j-1)*(2*N+1)+1:j*(2*N+1)),((m-1)*(2*N+1)+1:m*(2*N+1)) ,j,1)=   gradXbigM(:,:,m,j,1);
            gradXbig(((j-1)*(2*N+1)+1:j*(2*N+1)),((m-1)*(2*N+1)+1:m*(2*N+1)) ,j,2)=   gradXbigM(:,:,m,j,2);
            gradP_mj(:,:,m,j,1)= (k0*xjm*Dp_mj/absrjm(j,m) - (Dh_mj*yjm/((absrjm(j,m))^2)));
            gradP_mj(:,:,m,j,2)= (k0*yjm*Dp_mj/absrjm(j,m) + (Dh_mj*xjm/((absrjm(j,m))^2)));
            gradXbigM1(:,:,m,j,1) = - Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) *gradP_mj(:,:,m,j,1);
            gradXbigM1(:,:,m,j,2) = - Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) *gradP_mj(:,:,m,j,2) ; 
            gradXbig(((m-1)*(2*N+1)+1:m*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)),j,1)=   gradXbigM1(:,:,m,j,1);
            gradXbig(((m-1)*(2*N+1)+1:m*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)),j,2)=   gradXbigM1(:,:,m,j,2);
       end % ends if statement over cylinder      
   end
end
end
%%
%  s = '	SOLVING THE Linear system:   Bv = Xbig \  verAv '
Bv = Xbig \  verAv;   %       Xbig * BV = verCv ==> BV = inv(Xbig)* verCv
Bcoef = reshape(Bv, (2*N+1),M); % verCV is a colunm vector of length (2*N+1)*RN*M
%%
%  s = '	EVALUATING THE GRADIENT VECTOR COMPONENTS s_j'

C_nm=zeros((2*N+1),M);
    for I=1:numel(nv)
        n= nv(I);
        for m=1:M
        C_nm(I,m)=(-1i)^n *  exp(-1i*(k0)*absr(m)*cos(argr(m)));
        end
    end
    Cv=reshape(C_nm,M*(2*N+1),1);
%%
% Making Gradient Vectors
    gradcM= zeros((2*N+1),M,M,2);
    GradAinv=zeros((2*N+1),M,M,2);
    gradaM=zeros((2*N+1),M,M,2);
    gradAv=zeros((2*N+1)*M,M,2);
    gradCv=zeros((2*N+1)*M,M,2);
    for J=1:M
        J;
        for m=1:M
         m;
            if m==J
%                 gradcM(:,m,J,1)= -1i*k0/absr(m)*( XM(m)*cos(argr(m))+YM(m)*sin(argr(m)) )*C_n(:,m)
%                 gradcM(:,m,J,2)=  -1i*k0/absr(m)*( YM(m)*cos(argr(m))-XM(m)*sin(argr(m)) )*C_n(:,m)
                gradcM(:,m,J,1)=-1i*k0*C_nm(:,m);
                gradcM(:,m,J,2)=  0;
%                 gradcM(:,m,J,1)= -1i*k0*C_n(:,m);
%                 gradcM(:,m,J,2)=  0;
                GradAinv(:,m,J,1)= 1i*k0*Ainv(:,m);
                GradAinv(:,m,J,2)= 0; %1i*k0*Ainv(:,J);
            end          
        end
    end
   
    for I=1:2
        for J=1:M
        gradaM(:,:,J,I) = Tdiag( ((j-1)*(2*N+1)+1:j*(2*N+1)),((j-1)*(2*N+1)+1:j*(2*N+1)) ) * GradAinv(:,:,J,I);                  % gradAM is a  (2N+1)x M x M X 2  matrix
        gradAv(:,J,I)= reshape(gradaM(:,:,J,I), M*(2*N+1),1);
        gradCv(:,J,I)= reshape(gradcM(:,:,J,I), M*(2*N+1),1);
        end
    end       
    %%
% s = '	EVALUATING TSCS Q'    
%     f_n=zeros(1,2*N+1);
%     F_n=zeros(2*N+1,M);
%     g_n=zeros(2*N+1,M);
%     for nI=1:numel(nv)
%         n= nv(nI);
%         for m=1:M
%              F_n(nI,m)= Bcoef(nI,m) * exp(-1i*(k0)*absr(m)*cos(argr(m))); 
%              g_n(nI,m)=(-1i)^n * Bcoef(nI,m) * exp(-1i*(k0)*absr(m)*cos(argr(m)));     
%         end
%         FarG(Ifreq,nI) = sum(g_n(nI,:));
%         FarF(Ifreq,nI) = sum(F_n(nI,:)); 
%         f_n(nI)=2/(k0)*((-1i)^n)* FarF(Ifreq,nI);
%         f_n1(nI)=((-1i)^n)* FarF(Ifreq,nI);
%     end
    Q(Ifreq) = - 4/k0* real((Cv.') * Bv) ;
%%
% Making the Gradient Vectors s_j
% if nargout  > 1
%     s_j = zeros(M,2);
%     q_j = zeros(M,2);
    for I=1:2  % I=1 corresponds to e1 component(x) of gradient vector S_j, I=2 corresponds to e2 component(y) of gradient vectors 
%     I
        for J=1:M
%         J
            s_j(Ifreq,J,I)= -4/k0*real(  (gradCv(:,J,I)).'*Bv - (Cv.')*((Xbig)\(gradXbig(:,:,J,I)*Bv -gradAv(:,J,I)) ));
            q_j(Ifreq,J,I)=  Q(Ifreq)*s_j(Ifreq,J,I);
        end
    end
    
%  sV = reshape(s_j.',2*M,1)
%     qV= = reshape(s_j.',2*M,1);???
% 
% end

end
Q_RMS = sqrt ( (1/nfreq )*(sum(Q.^2)));
q_RMS_j = zeros(M,2);
q_RMS_j(:,:) = (1/ ((nfreq)*Q_RMS) ) * (sum (q_j,1)) ;
% kav=kav
qV= reshape( q_RMS_j.',2*M,1);

end
% toc
% q_j_sum=(sum (q_j,1))
% Q_RMS = sqrt ( (1/nfreq )*(sum(Q_square,1)))

% q_j_nfreq_1=(q_j(1,:,:))
% q_j_nfreq_2=(q_j(2,:,:))
% sum_q_j1=q_j_nfreq_1+q_j_nfreq_2
% sum_q_j(:,:) = (sum (q_j,1))
%      Q_RMS(Ifreq) = Q_RMS + ( Q(Ifreq))^2;
%     Q_square(Ifreq) = ( Q(Ifreq))^2;



