function [W,C]=initialization_for_quadsync(scaled_T)

% construct W
n = size(scaled_T,1)/3;
W = zeros(n,n,n,n);
for i = 1:n
    for j = 1:n
        for k = 1:n
            for l = 1:n
                if nnz(scaled_T(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l)) > 0
                    W(i,j,k,l) = 1;
                end
            end
        end
    end
end

% construct C
[U,S,sv] = mlsvd(scaled_T);
C = U{1}(:,1:4);

end




