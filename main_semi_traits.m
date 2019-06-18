function main_semi_traits(dataset)
% detection models with known parameters from species traits
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
    Z = importdata('dataset/Z6.csv',',');
    alpha = [0.1, 0.2, 0.1, 0.2, 0.1, 0.3].';
    R_list = [10,20,40];
elseif (dataset == "hpi") 
    N = importdata('dataset/host-parasite/Y.csv',',');
    A = NaN(size(N));
    A(N>0) = 1; 
    if with_nan
        N(isnan(A)) = nan; % let unknown interactions be NaN
    end
    Z = importdata('dataset/host-parasite/Z11.csv',',');
    alpha = [0.2, 0.05, 0.1, 0.01, 0.3, 0.07, 0.03, 0.2, 0.01, 0.1, 0.2].';    
    R_list = [5,10,15];
else
    error('Specify the existing detaset')
end
[I,J] = size(N);

%% 2nd Approach - generate detection probabilities from traits
% e.g., logit(p_ij) = 0.1 + 0.2*size_i + 0.3*size_j + 0.4*size_i*size_j.
[IJ,fea] = size(Z);
for i=1:fea
    Z(:,i) = Z(:,i)/max(Z(:,i));
end
P = Z*alpha;
%hist(P) % check the distribution
tt = max(P);
if tt > 1
    error('this trial is not useful')
end
P_mat = reshape(P,I,J);

%% sample from Binomial dist. to generate a new matrix Y
Y = binornd(N,P_mat); Y_sum = sum(Y(:));

%% find the best rank R
repeat_num = 10; 
max_iter = 100;

alpha_conv_list = zeros(length(R_list), max_iter);
alpha_est_list = zeros(length(R_list), fea);
for r = 1:length(R_list)
    R = R_list(r);
    [Y_est, alpha_conv, alpha_est] = learn_prob_mat(Y,Z,fea,alpha,R,...
        repeat_num,max_iter);
    alpha_conv_list(r,:) = alpha_conv;
    alpha_est_list(r,:) = alpha_est;
end

save(strcat('results/case1/semi_trait_alpha_conv_',dataset),'alpha_conv_list')

figure(2)
plot(alpha_conv_list','LineWidth',2.5); 
if (dataset == "ppi")
    leg = legend('F=10','F=20','F=40','Location','best');
elseif (dataset == "hpi")
    leg = legend('F=5','F=10','F=15','Location','best');
else
    error('Specify the existing detaset')
end
set(leg,'FontSize',15); set(gca,'FontSize',15);
xlabel('iterations','fontsize',16); ylabel('MSE','fontsize',16); 
title('Detection probability = function of features','fontsize',16)
print('-f2','-depsc',strcat('results/case1/semi_traits_',dataset))
print('-f2','-dpdf',strcat('results/case1/semi_traits_',dataset))
print('-f2','-dpng',strcat('results/case1/semi_traits_',dataset))