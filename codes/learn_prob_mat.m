function [Y_est,alpha_conv_mean,alpha_est_mean] = learn_prob_mat(Y,Z,fea,...
    alpha,R,repeat_num,max_iter)
% learn poisson N mixture model
% coded by Eugene Seo (seoe@oregonstate.edu)

[I,J] = size(Y);
alpha_conv_repeat = zeros(repeat_num,max_iter);
alpha_est_repeat = zeros(repeat_num,fea);
for m=1:repeat_num 
    disp(['R ', num2str(R), ', repeat ', num2str(m)])
    Init.U = rand(I,R);
    Init.V = rand(J,R);
    Init.alpha = rand(fea,1);
    [ U_est, V_est, alpha_est, result ] = prob_mat_nan(Y,Z,alpha,1,'MM_ITERS',...
        max_iter,'Initial',Init,'EXP_CODE', 1);                
    alpha_conv_repeat(m,:) = result.alpha_conv;
    alpha_est_repeat(m,:) = alpha_est;
    P_est = Z*alpha_est; % detection probability per pair        
    lambda_est = U_est*V_est'; % the average number of events        
    P_mat_est = reshape(P_est,I,J);
    Y_est = lambda_est.*P_mat_est;       
end
alpha_conv_mean = mean(alpha_conv_repeat);
alpha_est_mean = mean(alpha_est_repeat);