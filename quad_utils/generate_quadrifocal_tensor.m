function quad_tens = generate_quadrifocal_tensor(cams)
    n = size(cams,1)/3;
    quad_tens = zeros(3*n,3*n,3*n,3*n);
    
    for i =1:3*n
        for j = 1:3*n
            for k = 1:3*n
                for l = 1:3*n
                    quad_tens(i,j,k,l) = det([cams(i,:);cams(j,:);...
                        cams(k,:);cams(l,:)]);
                end
            end
        end
    end

end