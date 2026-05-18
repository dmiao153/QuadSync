function quad_tens = Quad_from_P(p1,p2,p3,p4)

    quad_tens = zeros(3,3,3,3);
    
    for i =1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    quad_tens(i,j,k,l) = det([p1(i,:);p2(j,:);...
                        p3(k,:);p4(l,:)]);
                end
            end
        end
    end

end