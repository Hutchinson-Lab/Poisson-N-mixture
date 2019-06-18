function [U,V,alpha,result] = prob_mat_nan(Y,Z,alpha_true,metric,varargin)
% Poisson N-mixture model with a probabilistic NMF
% This file can handle nan in Y; it is for testing the cases where we have
% unobserved entries in Y
% coded by Xiao Fu (xiao.fu@oregonstate.edu)

% Read the optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'EXP_CODE'
                exp_code = varargin{i+1};
            case 'U_TRUE'
                U_true = varargin{i+1};
            case 'V_TRUE'
                V_true = varargin{i+1};
            case 'Y_TRUE'
                Y_true = varargin{i+1};
            case 'IDX'
                idx = varargin{i+1};
            case 'MM_ITERS'
                MaxIt = varargin{i+1};
            case 'INITIAL'
                Init = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end    

% Set the defaults for the optional parameters
[I,J] = size(Y);
fea = size(Z,2);
U = Init.U;
V = Init.V;
alpha = Init.alpha;
innerMM = 20;
%time = 0;

true_nan_num = length(find(isnan(Y)));
for it = 1:MaxIt
    tic;
    if exp_code == 0
        disp(['running at iter ',num2str(it)])
    end

    lambda_mat = U*V';

    % =====  alpha updates ===        
    max_it_admm = 20;              
    [P_mat,alpha] = admm_p_sub(Y,Z,alpha,lambda_mat,max_it_admm);

    % ======== U,V updates ===        
    [nr,nc] = find(isnan(Y));
    nnr = find(isnan(Y));  
    if true_nan_num ~= length(nnr)
        error('error')
    end
    p_vec = P_mat(:);
    p_vec(nnr) = Z(nnr,:)*alpha;
    P_mat = reshape(p_vec,I,J); 

    for ii = 1:length(nr)
        Y(nr(ii),nc(ii)) = U(nr(ii),:)*V(nc(ii),:)';
    end 

    preU = U;
    for ii = 1:innerMM
        for i = 1:I
            tilde_U(i,:) = max(sum(diag(P_mat(i,:))*V),eps) ;
        end
        Phi = (Y./max(U*V',eps))*V;
        U = U.*Phi./tilde_U;
    end   

    preV = V;        
    for ii = 1:innerMM            
        for j=1:J
            tilde_V(j,:) = max(sum(diag(P_mat(:,j))*U),eps);
        end            
        PhiT = (Y'./max(V*U',eps))*U;
        V = V.*PhiT./tilde_V;
    end

    %time_ind=toc;
    %if it>1
    %    time(it)=time_ind + time(it-1);
    %else
    %    time(it)=time_ind;
    %end

    % obj(it)= -sum(sum(log((P_mat.*lambda_mat).^Y.*exp(-lambda_mat.*P_mat))));
    % if it>1&&abs(obj(it)-obj(it-1))<1e-3
    %   break;
    % end

    switch exp_code
        case 0
            result.MSE_UV(it) = 0.5*(MSE_measure(U_true,U)+MSE_measure(V_true,V));
            result.MSE_alpha(it) = (1/fea)*norm(alpha - alpha_true,2)^2;
        case 1
            result.alpha_conv(it) = sqrt((1/fea)*norm(alpha - alpha_true,2)^2);  
        case 2
            Y_est = prediction(U, V, Z, alpha, I, J);
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.valid);            
            result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
            RMSE_Y(it) = result.errors(metric);
            result.Y_est = Y_est; result.Iter = it-1;
            if it > 1 && RMSE_Y(it) > RMSE_Y(it-1)            
                Y_est = prediction(preU, preV, Z, alpha, I, J);
                [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.test);
                result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
                result.Y_est = Y_est; result.Iter = it-1;
                %disp(['error starts to increase at ', num2str(Iter)])
                break;
            end
        case 3
            result.Y_est = prediction(U, V, Z, alpha, I, J);
            result.N_est = U*V';
        otherwise
            disp('other value')
    end

    Y(nnr) = nan;
end