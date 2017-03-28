function result = matrixDist(data, dictionary)
% data: K * N
% dictionary: K * M
% result M * N
%% method 1
temp = bsxfun(@plus, sum(data.^2, 1), sum(dictionary.^2, 1)');
result = temp - 2*dictionary'*data;

% %% method 2 for comparison
% [K, N] = size(data);
% [K, M] = size(dictionary);
% result = zeros(M, N);
% for n = 1:N
%     error = sum(bsxfun(@minus, data(:, n), dictionary).^2, 1);
%     result(:, n) = error';
% end

end