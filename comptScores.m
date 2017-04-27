function score = comptScores(input, obj_weight)
    input = input<0.7;
    [wi,hi,ci] = size(input);
    [wo,ho,co] = size(obj_weight);
    assert(ci == co);
    if wi > wo
        diff1 = floor((wi-wo)/2);
        input = input(diff1+1:diff1+wo, :, :);
    elseif wi < wo
        diff1 = floor((wo-wi)/2);
        diff2 = wo-wi-diff1;
        input = padarray(input, [diff1 0 0], 0, 'pre');
        input = padarray(input, [diff2 0 0], 0, 'post');
    end
    assert(size(input, 1) == size(obj_weight,1));
    
    if hi > ho
        diff1 = floor((hi-ho)/2);
        input = input(:, diff1+1:diff1+ho, :);
    elseif hi < ho
        diff1 = floor((ho-hi)/2);
        diff2 = ho-hi-diff1;
        input = padarray(input, [0 diff1 0], 0, 'pre');
        input = padarray(input, [0 diff2 0], 0, 'post');
    end
    assert(size(input, 2) == size(obj_weight,2));
    
    term1 = sum(sum(sum(input .* obj_weight)));
    zero_m = zeros(size(obj_weight));
    term2 = -sum(logsumexp(cat(2, reshape(obj_weight, [], 1), reshape(zero_m, [], 1)), 2));
    score = term1 + term2;
    
end
