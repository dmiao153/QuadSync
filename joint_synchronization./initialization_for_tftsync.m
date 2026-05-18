function [W]=initialization_for_tftsync(scaled_T)

% construct W
n = size(scaled_T,1)/3;
W = zeros(n,n,n);
for i = 1:n
    for j = 1:n
        for k = 1:n
            if nnz(scaled_T(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k)) > 0
                W(i,j,k) = 1;
            end
        end
    end
end


end