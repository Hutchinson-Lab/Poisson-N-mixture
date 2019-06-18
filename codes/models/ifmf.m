function result = ifmf(Y_true,idx,Y,R,maxiter,exp_code,metric)
% Implicit Feedback Matrix Factorization (IFMF) model
% coded by Eugene Seo (seoe@oregonstate.edu)

[nPolls,nPlants] = size(Y);
lambda = 1;
alpha = 1;
W = 1 + alpha * Y; % confidence
U = rand(nPolls,R); 
V = rand(R, nPlants);
for it=1:maxiter
    oldU = U;
    oldV = V;
    for i=1:nPolls
        U(i,:) = inv(V*diag([W(i,:)]) * V' + ...
            diag(ones([1,R])*lambda)) * V * diag([W(i,:)])*Y(i,:)';
    end
    for j=1:nPlants
        V(:,j) = inv(U' * diag([W(:,j)]) * U + ...
        diag(ones([1,R]) * lambda)) * U' * diag([W(:,j)]) * Y(:,j);
    end
    U_diff = sqrt(sum(sum((oldU-U).^2)) / numel(U));
    V_diff = sqrt(sum(sum((oldV-V).^2)) / numel(V));
    %disp(['Iter ', num2str(it), ' mean alsolute changes in U:', ...
    %num2str(U_diff), ' and V: ', num2str(V_diff)])    

    Y_est = U*V;        
    switch exp_code
        case 2
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.test);
            result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
            result.Y_est = Y_est; result.Iter = it - 1;
            RMSE_Y(it) = result.errors(metric);
            if it > 1 && RMSE_Y(it) > RMSE_Y(it-1)            
                Y_est = oldU*oldV;
                [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.test);
                result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
                result.Y_est = Y_est; result.Iter = it - 1;                
                %disp(['error starts to increase at ', num2str(Iter)])
                break;
            end
        case 3
            result = Y_est;
        otherwise
            disp('other value')
    end  

    if U_diff + V_diff < 1e-6
        disp('break')
        break;
    end
end  