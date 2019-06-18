function result = colab_filtering_mod(X,F,metric,varargin)
%COLAB_FILTERING implements a colaborative filtering algorithm with missing
%values, the algorithm solves min ||X-AB'|| + lbd*(||A||+||B||), the least
%squares are calculated only the the position where observation is
%available. Solving the problem with Alternating Least Squares(ALS).
%   X: data matrix, with missing values represented by NaN
%   F: the length of the latent factors.
% coded by Xiao Fu (xiao.fu@oregonstate.edu)

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
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%   Adding bias to each user and item. 
lbd = 1;
mask = isnan(X); 
X0 = X;
X0(mask) = 0; 
mask = ~mask;
tot = sum(sum(mask));
[m,n] = size(X);
A = zeros(m,F);
B = randn(n,F);

crit = 1e-6;
ff = 2*crit;
it_tot = 100;
it = 0;
costold = 0;

% User and item bias
a = zeros(1,m)';
b = zeros(1,n)';
u = 0;

cost_it = zeros(1,it_tot);
while it < it_tot && ff > crit
    it  = it+1;
    %fprintf('Iteration: %d\n',it);
    %% Solving for A: user
    oldA = A;
    for i = 1:m
        Bi = B(mask(i,:),:);
        Xi = X(i,mask(i,:))'-u-a(i)-b(mask(i,:));
        XX = [Bi; sqrt(lbd)*eye(F)];
        yy = [Xi; zeros(F,1)];
        A(i,:) = (XX\yy)';
    end
    %% Solving for B: item
    oldB = B;
    for i = 1:n
        Ai = A(mask(:,i),:);
        Xi = X(mask(:,i),i)-u-b(i)-a(mask(:,i));
        XX = [Ai; sqrt(lbd)*eye(F)];
        yy = [Xi; zeros(F,1)];
        B(i,:) = (XX\yy)';
    end

    switch exp_code
        case 0
            result(it) = ...
                0.5*(MSE_measure(U_true,A) + MSE_measure(V_true,B));
        case 2
            Y_est = A*B';
            [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.test);
            result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
            RMSE_Y(it) = result.errors(metric);
            result.Y_est = Y_est; result.Iter = it - 1;            
            if it > 1 && RMSE_Y(it) > RMSE_Y(it-1)            
                Y_est = oldA*oldB';
                [RMSE,rRMSE,AUC_ROC,AUC_PR] = evaluation(Y_true, Y_est, idx.test);
                result.errors = [RMSE,rRMSE,AUC_ROC,AUC_PR];
                result.Y_est = Y_est; result.Iter = it - 1;                
                %disp(['error starts to increase at ', num2str(Iter)])
                break;
            end
        case 3
            result = A*B';
        otherwise
            disp('other value')
    end

    %% Update overall average
    u = sum(sum(mask.*(X0-A*B'-a*ones(1,n)-ones(m,1)*b')))/(tot + lbd);

    %% Update user bias
    for i=1:m
        num = sum(mask(i,:));
        Xi = X(i,mask(i,:))'-u-b(mask(i,:))-B(mask(i,:),:)*A(i,:)';
        yy = [Xi;zeros(num,1)];
        XX = [ones(num,1);sqrt(lbd)*ones(num,1)];
        a(i) = XX\yy;
    end
    %% Update item bias
    for i = 1:n
        num = sum(mask(:,i));
        Xi = X(mask(:,i),i)-u-a(mask(:,i))-A(mask(:,i),:)*B(i,:)';
        yy = [Xi;zeros(num,1)];
        XX = [ones(num,1);sqrt(lbd)*ones(num,1)];
        b(i) = XX\yy;
    end
    %% Calculating residuals
    cost = norm(mask.*(X0-u-a*ones(1,n)-ones(m,1)*b'-A*B'),'fro')^2 + ...
        lbd*(norm(A,'fro')^2+norm(B,'fro')^2+norm(a)^2+norm(b)^2 + u^2);
    cost_it(it) = cost;
    ff = abs(cost-costold)/cost;
    costold = cost;
end
%plot(1:it,cost_it(1:it))