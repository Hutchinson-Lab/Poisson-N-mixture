function Y_est = prediction(U_est,V_est,Z,alpha_est,I,J)
% predict the observed Y
% coded by Eugene Seo (seoe@oregonstate.edu)

N_est = U_est*V_est';
P_vec = Z*alpha_est;
P_mat = reshape(P_vec,I,J);
P_mat = min(P_mat,1);    
P_mat = max(P_mat,0);
if min(P_mat(:)) < 0 || max(P_mat(:)) > 1
    disp(num2str(min(P_vec)))
    error('no')
end
Y_est = N_est.*P_mat;