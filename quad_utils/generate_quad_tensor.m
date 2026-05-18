function quad_tens = generate_quad_tensor(np,n)
    cams = zeros(3*n,4);
    for i = 1:n    
        cams(3*(i-1)+1:3*i,:) = np{i};
    end

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