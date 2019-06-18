% this script offers a simple demo for the proposed approach
% coded by Xiao Fu (xiao.fu@oregonstate.edu)
% A Possion-Binomial probablistic decomposition framework for 
% ecosystem link prediction. this file contains "nan" type of data in Y

clear; clc; close all; addpath(genpath(pwd))
%% parameter setup
I = 50;
J = 50; 
R = 8; % assume that the rank of the matrix is R
lambda = zeros(I,J);
TT = 3;
maxiter = 100;

for trial = 1:TT    
    %% data generation
    % generate embedding features U and V
    U = rand(I,R);
    U(1:R,1:R) = eye(R);    
    V = rand(J,R);
    V(1:R,1:R) = eye(R);   
    beta = 15;
    U = beta*U;
    V = beta*V;   
    
    % generate the true abundance
    lambda_mat = U*V';
    N = poissrnd(lambda_mat,I,J);
    
    % generate detection probability feature Z
    fea = 8; % number of features per observation unit
    alpha = rand(fea,1);
    Z = 1*rand(I*J,fea);
    for m = 1:I*J
        Z(m,:) = Z(m,:)/sum(Z(m,:));
    end
    
    % generate the detection probability
    P_vec = Z * alpha;
    tt = max(P_vec);
    P_vec = P_vec / tt;
    alpha = alpha / tt;  
    if tt > 1
        error('this trial is not useful')
    end    
    P_mat = reshape(P_vec,I,J);
     
    % generate observations
    Y = zeros(I,J);
    for i=1:I
        for j=1:J
            % the binomial distribution with parameters n and p
            Y(i,j) = binornd(N(i,j), P_mat(i,j)); 
        end
    end    
    if nnz(sum(Y)) < size(Y,2) || nnz(sum(Y,2)) < size(Y,1)
        error('this trial is not useful')
    end    
    
    % make missing entries
    Y(1,1) = nan;
    Y(3,2) = nan;
    
    %% algorithm: the algorithm aims at finding out U, V, and alpha    
    Init.U = rand(I,R);
    Init.V = rand(J,R);
    Init.alpha = rand(fea,1);
 
    % learn Poisson NMF
    [~,~,~,result] = ...
        poisson_mat_nan(Y,Z,alpha,1,'U_TRUE',U,'V_TRUE',V,'MM_ITERS',maxiter,...
        'INITIAL',Init,'EXP_CODE',0);
    MSE_UV_Poisson(trial,:) = result.MSE_UV;
    %MSE_alpha_Poisson(trial,:) = result.MSE_alpha;
    
    % learn Poisson N-mixture NMF
    [~,~,~,result] = ...
        prob_mat_nan(Y,Z,alpha,1,'U_TRUE',U,'V_TRUE',V,'MM_ITERS',maxiter,...
        'INITIAL',Init,'EXP_CODE',0);
    MSE_UV_proposed(trial,:) = result.MSE_UV;
    MSE_alpha_proposed(trial,:) = result.MSE_alpha;
    X = Y;    
    [MSE_UV_CF(trial,:)] = colab_filtering_mod(X,R,1,'U_TRUE',U,'V_TRUE',V,...
        'EXP_CODE',0);
end

%% results
figure(1)
semilogy(1:maxiter,mean(MSE_UV_Poisson),'-b*'); hold on
semilogy(1:maxiter,mean(MSE_UV_proposed),'-rs'); hold on
semilogy(1:maxiter,mean(MSE_UV_CF),'-kd'); hold on
legend('Poisson NMF','Proposed','MC-CF')
xlabel('iterations')
ylabel('MSE')
set(gca,'fontsize',14,'linewidth',2)
print('-depsc','results/sim_iter_vs_acc')

figure(2)
semilogy(1:maxiter,mean(MSE_alpha_proposed),'-rs'); hold on
legend('Proposed')
xlabel('iterations')
ylabel('MSE')
set(gca,'fontsize',14,'linewidth',2)
print('-depsc','results/sim_itervsalphaacc')