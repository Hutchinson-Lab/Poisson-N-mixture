function [F] = test_sample_cross_validation(N, fold_num, A)       
    if false % only positive
        list_pos_int = find(N > 0);
        shuffled_pos_list = list_pos_int(randperm(length(list_pos_int)));
        sample_num_pos = floor(length(list_pos_int) / fold_num);
        shuffled_pos_list = shuffled_pos_list(1:sample_num_pos*fold_num);
        test_idx = reshape(shuffled_pos_list, fold_num, sample_num_pos);
    else % both positive and negative
        F = zeros(size(N));
        indicator_list = [0,1];
        for i = 1:length(indicator_list)
            indicator = indicator_list(i);
            list_int = find(A == indicator);
            shuffled_list = list_int(randperm(length(list_int)));
            sample_num = floor(length(list_int) / fold_num);
            shuffled_first_list = shuffled_list(1:sample_num*(fold_num-1));
            shuffled_remain_list = shuffled_list(sample_num*(fold_num-1)+1:length(shuffled_list));
            test_idx = reshape(shuffled_first_list, fold_num-1, sample_num);
            for j=1:size(test_idx,1)
                idx = test_idx(j,:);
                if sum(F(idx)) > 0
                    disp('error')
                end                
                F(idx) = j;
            end
            F(shuffled_remain_list) = fold_num;        
        end
    end    
end