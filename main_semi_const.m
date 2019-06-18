function main_semi_const(dataset)
% detection models with a known parameter, constant detection probability
% coded by Eugene Seo (seoe@oregonstate.edu)

addpath(genpath(pwd))
%% load N (observed data) and Z (features)
with_nan = true;
if (dataset == "ppi") 
    N = importdata('dataset/Y.csv',',');
    A = importdata('dataset/A.csv',',');
    if with_nan
        N(isnan(A)) = nan; % let unknown interactions be NaN
    end
    R_list = [10,20,40];
elseif (dataset == "hpi") 
    N = importdata('dataset/host-parasite/Y.csv',',');
    A = NaN(size(N));
    A(N>0) = 1; 
    if with_nan
        N(isnan(A)) = nan; % let unknown interactions be NaN
    end
    R_list = [5,10,15];
else
    error('Specify the existing detaset')
end
[I,J] = size(N);

%% 1st approach - Use a constant detection probability
const_P = 0.9;
alpha = const_P.'; Z = ones(numel(N), 1);
P = Z * alpha; 
tt = max(P);
if tt > 1
    error('this trial is not useful')
end
P_mat = reshape(P, I, J);
[~,fea] = size(Z); 
Y = binornd(N,P_mat); Y_sum = sum(Y(:));

%% find the best rank R
repeat_num = 10; 
max_iter = 100;

alpha_conv_list = zeros(length(R_list),max_iter);
alpha_est_list = zeros(length(R_list), 1);
for r = 1:length(R_list)
    R = R_list(r);
    [Y_est, alpha_conv, alpha_est] = learn_prob_mat(Y, Z, fea, alpha, R,...
        repeat_num, max_iter);  
    alpha_conv_list(r,:) = alpha_conv;
    alpha_est_list(r,:) = alpha_est;
end

save(strcat('results/case1/semi_const_alpha_conv_',dataset),'alpha_conv_list')

figure(1)
if (dataset == "ppi")
    plot(alpha_conv_list','LineWidth',2.5); 
    leg = legend('F=10','F=20','F=40','Location','northwest');
elseif (dataset == "hpi")
    plot(alpha_conv_list','LineWidth',2.5); 
    leg = legend('F=5','F=10','F=15','Location','northwest');
else
end
set(leg,'FontSize',15); set(gca,'FontSize',15);
xlabel('iterations','fontsize',16); ylabel('MSE','fontsize',16); 
title(['Constant detection probability = ',num2str(const_P)],'fontsize',16)
print('-f1','-depsc',strcat('results/case1/semi_const_',dataset))
print('-f1','-dpdf',strcat('results/case1/semi_const_',dataset))
print('-f1','-dpng',strcat('results/case1/semi_const_',dataset))