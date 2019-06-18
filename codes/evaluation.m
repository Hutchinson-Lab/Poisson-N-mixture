function [RMSE,rRMSE,AUROC,AUPRC] = evaluation(Y_true,Y_est,test_idx)
% calculate the errors between true Y and predicted Y
% coded by Eugene Seo (seoe@oregonstate.edu)

%% RMSE
diff = (Y_true(test_idx)-Y_est(test_idx)).^2;
RMSE = sqrt(sum(diff(:))/length(diff));    
rRMSE = RMSE/mean(Y_true(test_idx));
%fRMSE = RMSE/( max(Y_true(test_idx)) - min(Y_true(test_idx)) ); 

%% AUC-ROC & AUC-PR
ground_truth = false(length(test_idx),1);
ground_truth(Y_true(test_idx) > 0) = 1;
if length(unique(ground_truth)) == 1 % When test set has only positive data
    version = 1;    
    if version == 1 % UROC/AUPRC with unknown = 0
        % scope of recommendation a fold's test set + NaNs
        neg_idx = isnan(Y_true);
        tmp = NaN(size(Y_est));
        tmp(neg_idx) = 0;
        tmp(test_idx) = 1;
        remove_idx = find(isnan(tmp)); % validation set should not be targets
        y_true = tmp(:);
        y_score = Y_est(:);
        y_true(remove_idx) = [];
        y_score(remove_idx) = [];
        y_score(isnan(y_score)) = 0;
        [~,~,~,AUROC] = perfcurve(y_true,y_score,1);
        [~,~,~,AUPRC] = perfcurve(y_true,y_score,1,'xCrit', 'reca', 'yCrit', 'prec');    
    elseif version == 2 % AUROC/AUPRC with top K recommendations  
        nan_idx = find(isnan(Y_true));
        rank_idx = [test_idx; nan_idx];
        no_rank_idx = 1:(size(Y_true,1)*size(Y_true,2));
        no_rank_idx(rank_idx) = [];
        R = rank_by_row(Y_est,no_rank_idx);
        compute_PR(R, test_idx, 10);
        commandStr = 'python codes/compute_auprc.py';
        [~, commandOut] = system(commandStr);
        AUPRC = str2double(commandOut);
    elseif version == 4 % HR
        nan_idx = find(isnan(Y_true));
        rank_idx = [test_idx; nan_idx];
        no_rank_idx = 1:(size(Y_true,1)*size(Y_true,2));
        no_rank_idx(rank_idx) = [];
        R = rank_by_row(Y_est,no_rank_idx);
        topK = 1;
        AUPRC = compute_HR(R, test_idx, topK);
    elseif version == 5 % NDCG
        nan_idx = find(isnan(Y_true));
        rank_idx = [test_idx; nan_idx];
        no_rank_idx = 1:(size(Y_true,1)*size(Y_true,2));
        no_rank_idx(rank_idx) = [];
        R = rank_by_whole(Y_est,no_rank_idx);
        AUPRC = compute_NDCG(R, test_idx);
    elseif version == 6 % Correlation
        AUPRC = compute_corr(Y_true, Y_est, test_idx);
    else
        AUPRC = 0;
    end    
    
else % When test set has both positive & negative data
    y_score = Y_est(test_idx);
    [~,~,~,AUROC] = perfcurve(ground_truth,y_score,'true');  
    [~,~,~,AUPRC] = perfcurve(ground_truth,y_score,'true','xCrit', 'reca', 'yCrit', 'prec');
end

function R = rank_by_row(Y_est,no_rank_idx)
    Y_est(no_rank_idx) = min(Y_est(:))-10000;
    R = zeros(size(Y_est));
    for i=1:size(Y_est,1)
        X = Y_est(i,:);
        [~,X_ranked]  = ismember(X,flip(unique(X)));
        R(i,:) = X_ranked;
    end
    R(no_rank_idx) = length(Y_est(:))+1;

function R = rank_by_whole(Y_est,no_rank_idx)
    R = zeros(size(Y_est));
    Y_est(no_rank_idx) = min(Y_est(:))-10000;
    X = Y_est(:);
    [~,X_ranked]  = ismember(X,flip(unique(X)));
    R(:) = X_ranked;
    R(no_rank_idx) = length(Y_est(:))+1;

function compute_PR(R, test_idx, maxK)
    PR = zeros(maxK+1,2);
    pos_num = length(test_idx);
    for k=1:maxK
        pred_idx = find(R <= k);
        correct_num = length(intersect(pred_idx,test_idx));
        predic_num = length(pred_idx);
        PR(k,1) = correct_num / predic_num; % precision
        PR(k,2) = correct_num / pos_num; % recall
    end
    PR(k+1,1) = 0;
    PR(k+1,2) = 1;
    csvwrite('results/PR.csv', PR)

function HR = compute_HR(R, test_idx, topK)
    T = zeros(size(R));
    P = zeros(size(R));
    pred_idx = R <= topK;
    P(pred_idx) = 1;
    T(test_idx) = 1;
    score = P .* T;
    HR = sum(score(:));

function NDCG = compute_NDCG(R, test_idx)
    test_ranks = R(test_idx);
    NDCG = sum(1./log2(test_ranks+1));
    
function CORR = compute_corr(Y_true, Y_est, test_idx)
    [~,true_rank] = sort(-Y_true(test_idx));
    [~,pred_rank] = sort(-Y_est(test_idx));
    %R = corrcoef(true_rank, pred_rank, 'alpha', 0.05);    
    %CORR = R(1,2);
    CORR = corr(true_rank, pred_rank,'Type','Spearman');