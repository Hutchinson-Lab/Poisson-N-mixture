function main_folds(dataset, metric)
% detection models with real datasets using cross-validation folds
% coded by Eugene Seo (seoe@oregonstate.edu)

addpath(genpath(pwd))
%% setup parameters
with_nan = true;

%% load datasets
if (dataset == "ppi") 
    N = importdata('dataset/Y.csv',',');
    A = importdata('dataset/A.csv',',');    
    N(isnan(A)) = nan; % let unknown interactions be NaN
    Zid = 3;
    R_list = [2,5,10,20,40];
elseif (dataset == "hpi") 
    N = importdata('dataset/host-parasite/Y.csv',',');
    A = zeros(size(N));
    A(N>0) = 1;     
    Zid = 4;
    R_list = [2,5,10,15];
else
    error('Specify the existing detaset')
end

[Z, alpha] = define_Z(N, Zid);
idx_known = find(~isnan(N));

%% sampling test sets
if (dataset == "ppi") fold_num = 10; else fold_num = 10; end 

file_path = strcat('dataset/folds-',dataset,'-',num2str(size(N,1)),'x',num2str(size(A,2)),'.mat');
if isfile(file_path)
    disp('loading folds')
    load(file_path);
else
    disp('saving folds')
    [folds] = test_sample_cross_validation(N, fold_num, A);
    save(file_path, 'folds')
end

%% Learn the model
model_num = 5; repeat_num = 10; write = false;

whole_repeat = 1;
rep_summary = zeros(model_num,whole_repeat);
rep_combined_summary = zeros(model_num,whole_repeat);
for rep=1:whole_repeat
    best_summary = zeros(model_num,6); % RMSE, AUC_RPC, AUC_PR, Iter, minRank
    summary_combined = zeros(model_num,6); % RMSE, AUC_RPC, AUC_PR, minRank, Iter
    Start = tic;
    for m=1:model_num
        mStart = tic;
        rank_summary = zeros(length(R_list),5);
        rank_summary_combined = zeros(length(R_list),4);
        for r=1:length(R_list)
            rStart = tic;
            R = R_list(r);
            fold_summary = zeros(fold_num,5);
            Yest_combined = nan(size(N));
            for f=1:fold_num
                disp(['Repeat: ', num2str(rep), ' Model: ', num2str(m), ', Rank: ', num2str(R), ', Fold: ', ...
                    num2str(f)])
                idx = get_folds_idx(folds, f); Ynan = N;
                if with_nan
                    Ynan(idx.test) = nan;
                    Ynan(idx.valid) = nan;
                end
                %nnr = find(isnan(Ynan(:)));
                [min_eval,pred] = learn_models_folds(N, Ynan, idx, Z, alpha, ...
                    R, m, repeat_num, write, f, metric);
                fold_summary(f,:) = min_eval;
                if write
                    csvwrite(['Y_est/4.M', num2str(m), '_R', num2str(R), '_F', ...
                        num2str(f), '.csv'], pred) 
                end
                Yest_combined(idx.test) = pred;   
            end        
            if write
                csvwrite(['results/3.fold_summary_M', num2str(m), '_R', ...
                    num2str(R), '.csv'], fold_summary) 
                csvwrite(['results/3.combinedYest_M', num2str(m), '_R', ...
                    num2str(R), '.csv'], Yest_combined)             
            end 
            rank_summary(r,:) = mean(fold_summary); 
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(N, Yest_combined, idx_known);
            rank_summary_combined(r,:) = [RMSE,rRMSE,AUC_ROC,AUC_PR]; 
            rElapsed = toc(rStart);
            disp(['Total time per rank ', num2str(rElapsed)])
        end
        if write            
            csvwrite(['results/2.rank_summary_M',num2str(m),'.csv'], rank_summary) 
            csvwrite(['results/3.rank_combined_summary_M',num2str(m),'.csv'], ...
                rank_summary_combined) 
        end    
        if metric == 1 || metric == 2 
            min_rank = rank_summary(:,metric) == min(rank_summary(:,metric));
            min_rank2 = rank_summary_combined(:,metric) == min(rank_summary_combined(:,metric));
        else
            min_rank = rank_summary(:,metric) == max(rank_summary(:,metric));
            min_rank2 = rank_summary_combined(:,metric) == max(rank_summary_combined(:,metric));
        end

        if length(min_rank2) > 1 % when there more more than one min
            tmp = false(1,4);
            true_idxs = find(min_rank == true);
            tmp(true_idxs(1)) = true;
            min_rank= tmp;
        end
        best_summary(m,1:5) = rank_summary(min_rank,:);
        best_summary(m,6) = R_list(min_rank); 

        if length(min_rank2) > 1 % when there more more than one min
            tmp = false(1,4);
            true_idxs = find(min_rank2 == true);
            tmp(true_idxs(1)) = true;
            min_rank2= tmp;
        end
        summary_combined(m,1:4) = rank_summary_combined(min_rank2,:);           
        summary_combined(m,5) = best_summary(m,5);
        summary_combined(m,6) = R_list(min_rank2);
        mElapsed = toc(mStart);
        disp(['Total time per model ', num2str(mElapsed)])
    end

    column = {'Method';'Pois. N-mix';'Pois. NMF';'IFMF';'MC-CF';'Trunc. SVD'};
    %if (dataset == "ppi") 
        rep_combined_summary(:,rep) = summary_combined(:,metric);
        header = {'RMSE','rRMSE','AUROC','AUPRC','Iter','minRank'};        
        summary_combined = [header; num2cell(summary_combined)];
        summary_combined = [column, summary_combined]
        filename = strcat('results/case2/',dataset,'/combined_',header(metric),'.csv');
        writetable(cell2table(summary_combined),filename,'WriteVariableNames',false) 
    %else
        rep_summary(:,rep) = best_summary(:,metric)
        header = {'RMSE','rRMSE','AUROC','AUPRC','iter','minRank'};
        best_summary = [header; num2cell(best_summary)];
        best_summary = [column, best_summary];
        filename = strcat('results/case2/',dataset,'/foldavg_',header(metric),'_',num2str(rep),'.csv');
        writetable(cell2table(best_summary),filename,'WriteVariableNames',false) 
        Elapsed = toc(Start);
        disp(['Total time ', num2str(Elapsed)])
    %end
end

%csvwrite(strcat('results/case2/',dataset,'/rep_combined_summary.csv'), rep_combined_summary);
%csvwrite(strcat('results/case2/',dataset,'/rep_summary.csv'), rep_summary);