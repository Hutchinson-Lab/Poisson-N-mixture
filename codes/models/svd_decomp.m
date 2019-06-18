function result = svd_decomp(Y_true,idx,Ynan,R,exp_code)
% truncated singular value decomposition (SVD) model
% coded by Eugene Seo (seoe@oregonstate.edu)
 
[U,S,V] = svd(Ynan);
Y_est = U(:,1:R) * S(1:R,1:R) * V(1:R,:);    
switch exp_code
    case 2
        [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.test);   
        result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
        result.Y_est = Y_est;
    case 3
        result = Y_est;
    otherwise
        disp('other value')
end        