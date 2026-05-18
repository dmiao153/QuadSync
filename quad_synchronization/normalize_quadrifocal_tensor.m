function nqt = normalize_quadrifocal_tensor(quadrifocal_tensor)
    n = size(quadrifocal_tensor,1) / 3;
    nqt = zeros(3*n,3*n,3*n,3*n);

    for i = 1:n
        for j = 1:n
            for k = 1:n
                for l = 1:n
                    if nnz(quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l)) > 0
                        cur_ten = quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l);
                        nqt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l) = cur_ten / norm(cur_ten(:));
                    end
                end
            end
        end
    end

end