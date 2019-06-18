function [min_eval,pred] = learn_models_folds(N,Ynan,idx,Z,alpha,R,...
    model,repeat_num,write,test_fold,metric)
% learn all models from observed Y
% coded by Eugene Seo (seoe@oregonstate.edu)

nnr = find(isnan(Ynan(:)));
maxiter = 100;
[I,J] = size(Ynan); [~,fea] = size(Z);
if model == 5 % SVD
    Ynan(isnan(Ynan)) = 0;
    result = svd_decomp(N,idx, Ynan, R, 2);   
    Y_est = result.Y_est;
    t = num2cell(result.errors);
    [RMSE,rRMSE,AUC_ROC,AUC_PR] = deal(t{:});
    min_eval = [RMSE,rRMSE,AUC_ROC,AUC_PR,1];
    pred = Y_est(idx.test);
else
    eval_repeat = zeros(repeat_num,5);    
    Yest_repeat = zeros(repeat_num,I*J);   
    for rep=1:repeat_num
        Init.U = rand(I,R);
        Init.V  = rand(J,R);
        Init.alpha = normrnd(0,1,[fea,1]); %rand(fea,1);        
        if model == 1 % Proposed
            [~,~,alpha,result] = prob_mat_nan(Ynan,Z,alpha,metric,'Y_TRUE', ...
                N,'IDX',idx,'MM_ITERS',maxiter,'INITIAL',Init,...
                'EXP_CODE',2);
            Y_est = result.Y_est; Iter = result.Iter;
            t = num2cell(result.errors);
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = deal(t{:});
        elseif model == 2 % Poisson NMF
            [~,~,alpha,result] = poisson_mat_nan(Ynan,Z,alpha,metric,'Y_TRUE',...
               N,'IDX',idx,'MM_ITERS',maxiter,'INITIAL',Init,'EXP_CODE',2);
            Y_est = result.Y_est; Iter = result.Iter;
            t = num2cell(result.errors);
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = deal(t{:});
        elseif model == 3 % IFMF  
            Ynan(isnan(Ynan)) = 0; 
            result = ifmf(N,idx, Ynan, R, maxiter, 2, metric);       
            Y_est = result.Y_est; Iter = result.Iter;
            t = num2cell(result.errors);
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = deal(t{:});
        elseif model == 4 % CF
            %Ynan(isnan(Ynan)) = 0; 
            result = colab_filtering_mod( Ynan,R,metric,'Y_TRUE',N,'IDX',idx,...
                'EXP_CODE',2);  
            Y_est = result.Y_est; Iter = result.Iter;
            t = num2cell(result.errors);
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = deal(t{:});
        else 
            disp('no model')
            break;
        end
        eval_repeat(rep,:) = [RMSE,rRMSE,AUC_ROC,AUC_PR,Iter];
        Yest_repeat(rep,:) = Y_est(:);
        if write
            csvwrite(['Y_est/tmp/5.M',num2str(model),'_R',num2str(R),'_F',...
                num2str(test_fold),'_P',num2str(rep),'.csv'],Y_est) 
        end             
    end
    if metric == 1 || metric == 2 
        min_idx = find(eval_repeat(:,metric) == min(eval_repeat(:,metric)));
    else
        min_idx = find(eval_repeat(:,metric) == max(eval_repeat(:,metric)));
    end
    min_eval = eval_repeat(min_idx,:);        
    pred = Yest_repeat(min_idx,idx.test);
    if length(min_idx) > 1
        min_eval = min_eval(1,:);
        pred = Yest_repeat(min_idx(1),idx.test);
    end
end    