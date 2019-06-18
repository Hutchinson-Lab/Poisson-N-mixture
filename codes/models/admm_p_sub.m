function [P_mat,alpha]= admm_p_sub(Y,Z,alpha,lambda_mat,max_it_admm)   
% Alternating Direction Method of Multipliers (ADMM) based algorithm
% coded by Xiao Fu (xiao.fu@oregonstate.edu)

[I,J] = size(lambda_mat); 

%%% NaN handling
[nnr,~] = find(isnan(Y(:)));
Z(nnr,:) = [];
Y_vec = Y(:);
Y_vec(nnr) = [];    
omega = zeros(I*J,1);
omega(nnr) = [];
lambda_vec = lambda_mat(:);
lambda_vec(nnr) = [];
p_index = 1:I*J;
p_index(nnr) = []; 
%%% NaN handling

rho = sqrt((Y_vec'*Y_vec)/(I*J));
ZZinv = inv(Z'*Z);    

for it = 1:max_it_admm        
    p_bar = (Z*alpha - omega);
    rho_temp = (rho*p_bar - lambda_vec);
    p_temp = (((rho_temp) + sqrt(rho_temp.^2 + 4*rho*Y_vec)))/(2*rho);
    p = min(max(p_temp,0),1);

    % alpha_old = alpha;
    alpha = ZZinv*Z'*(p+omega);
    omega = p - Z*alpha + omega;

    % check convergence
    % tol = 1e-8;
    % mu = 10;
    % r = sum((p-Z*alpha).^2); % primal residual
    % s = sum(rho*Z*(alpha - alpha_old).^2);
    % if r+s <= tol
    %     break;
    % end
    % if r>mu*s
    %    rho = 2*rho;
    % elseif s>mu*r
    %     rho=rho/2;
    % else rho=rho;
    % end
end

p_large = nan(I*J,1);
p_large(p_index) = p;
P_mat = reshape(p_large,I,J);