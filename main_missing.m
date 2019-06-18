% predict missing entries in the real datasets
% coded by Eugene Seo (seoe@oregonstate.edu)

clear;clc;close all; addpath(genpath(pwd))
%% load datasets
Y_true = importdata('dataset/Y.csv',','); [I,J] = size(Y_true);
A = importdata('dataset/A.csv',','); % Availability matrix
Y_true(isnan(A)) = nan; % let unknown interactions NaN
idx_known = find(~isnan(Y_true));

%% load Z
Zid = 3; % Z type: 1) const Z, 2) biomass traits 3) picked by Justin
[Z, alpha] = define_Z(Y_true, Zid); [~,fea] = size(Z);

%% learn the model
R_per_model = [5,10,5,2,2]; 
Iter_per_model = [3,4,100,92,1];
write = false; repeat_num = 10; 
for repeat = 1:repeat_num
    disp(['repeat: ', num2str(repeat)])    
    for m=1:1
        mStart = tic;
        R = R_per_model(m);
        maxiter = Iter_per_model(m);         
        idx.test = find(isnan(A));
        [N_est, Y_est] = learn_models_missing(Y_true, Y_true, idx, Z, alpha, ...
            R, m, maxiter);
        if write
            csvwrite(['Y_est/Case3_Nest_',num2str(repeat),'.csv'], N_est) 
            csvwrite(['Y_est/Case3_Yest_',num2str(repeat),'.csv'], Y_est) 
        end
    end
end