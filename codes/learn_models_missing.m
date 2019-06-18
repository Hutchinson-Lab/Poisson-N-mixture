function [N_est,Y_est] = learn_models_missing(Y_true,Ynan,idx,Z,alpha,R,...
    model,maxiter)
% learn all models from observed Y
% coded by Eugene Seo (seoe@oregonstate.edu)

[I,J] = size(Ynan); [~,fea] = size(Z);
N_est = 1;
Init.U = rand(I,R);
Init.V  = rand(J,R);
Init.alpha =  rand(fea,1);   
switch model        
    case 1 % Proposed
        [~,~,~,result] = prob_mat_nan(Ynan,Z,alpha,1,'Y_TRUE',Y_true,'IDX',...
            idx,'MM_ITERS',maxiter,'Initial',Init,'EXP_CODE',3);
        N_est = result.N_est;
        Y_est = result.Y_est;
    case 2 % Poisson NMF
        Y_est = poisson_mat_nan(Ynan,Z,alpha,1,'Y_TRUE',Y_true,...
            'IDX',idx,'MM_ITERS',maxiter,'INITIAL',Init,'EXP_CODE',3);
    case 3 % IFMF
        Ynan(isnan(Ynan)) = 0; 
        Y_est = IFMF3(Y_true,idx, Ynan, R, maxiter, 3);        
    case 4 % CF
        Ynan(isnan(Ynan)) = 0; 
        Y_est = colab_filtering_mod(Ynan,R,1,'Y_TRUE',Y_true,'IDX',idx,...
            'EXP_CODE',3); 
    case 5
        Ynan(isnan(Ynan)) = 0; 
        Y_est = svd_decomp(Y_true, idx, Ynan, R, 3);             
    otherwise 
        disp('other value')
end  