function [W]=initialization_for_esync(scaled_T)

    % construct W
    n = size(scaled_T,1)/3;
    W = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if nnz(scaled_T(3*(i-1)+1:3*i,3*(j-1)+1:3*j)) > 0
                W(i,j) = 1;
            end
        end
    end


end