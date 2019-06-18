function [Z,alpha] = define_Z(Y,Zid)  
% coded by Eugene Seo (seoe@oregonstate.edu)

if Zid == 1 % 1. const Z    
    Z = ones(numel(Y), 1); alpha = 0.5;
elseif Zid == 2 % 2. our fearue Z with defined alpha    
    Z = importdata('dataset/Z3.csv',','); alpha = [0.2, 0.9, 0.5].';
elseif Zid == 3 %  3. Justin's feature
    Z = csvread('dataset/Z6.csv'); 
    alpha = [0.1, 0.2, 0.1, 0.2, 0.1, 0.3].';
elseif Zid == 4
    Z = csvread('dataset/host-parasite/Z11.csv'); 
    alpha = [0.04, 0.1, 0.09, 0.03, 0.3, 0.04, 0.01, 0.2, 0.01, 0.2, 0.07].';
else 
    disp('not yet')
end       

[~,fea] = size(Z);
for i=1:fea
    Z(:,i) = Z(:,i)/max(Z(:,i));
end

P = Z*alpha;
if max(P) > 1
    error('this trial is not useful')
end