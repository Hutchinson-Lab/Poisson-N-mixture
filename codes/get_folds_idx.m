function idx = get_folds_idx(folds,test_fold)
% retrieve the corresponding entries according to the given fold
% coded by Eugene Seo (seoe@oregonstate.edu)

if test_fold < 1 || test_fold > 10
    error('this test_fold is wrong')
end

fold_list = 1:max(folds(:));
if test_fold == 1
    valid_fold = max(folds(:));
else
    valid_fold = test_fold-1;
end
train_folds = fold_list;
train_folds([test_fold, valid_fold]) = [];

idx.test = find(folds == test_fold);
idx.valid = find(folds == valid_fold);
idx.train = find(ismember(folds, train_folds));    