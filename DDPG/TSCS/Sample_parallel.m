function [ Effs ] = Eval_Eff_1D_parallel(imgs, a, aa, M, kamax, kamin, nfreq, N)

Effs = zeros(1, N);
imgs = squeeze(imgs);
tic
% It takes 100 batches of images and executes them in a pararrel loop. 
% For more information, please see the link below
% https://www.mathworks.com/help/parallel-computing/parfor.html;jsessionid=0263486711e6bf7a2ed498e3642f
parfor n = 1:N
	img = imgs(n, :);
	Effs(n) = objectiveFunctionTSCS_RMS_kamin_kamax(img, a, aa, M, kamax, kamin, nfreq)
end
toc
end