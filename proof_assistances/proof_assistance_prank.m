clear all;
s = rng;
rng(1)

n = 15;
% Here we check the Prank for the uncalibrated case. 
[U,V,np,F,t] = generate_random_projection_cameras(n,false,false); 
% 2nd parameter outputs the camera locations. 3rd parameter is true for
% uncalibrated, false for calibrated. 
% output F is the block fundamental matrix. 

Q = generate_quad_tensor(np,n);






proj1 = zeros(3*n,3*n);
X = rand(3*n,1) * rand(1,3*n);
for i = 1:3*n
    for j = 1:3*n
        proj1 = proj1 + X(i,j) * reshape(Q(i,j,:,:),[3*n,3*n]);
    end
end

C = reorder_np_cell_to_array(np);
G = generate_quad_core();

prodtens = tmprod(G,{C,C,C,C},1:4);
diff = prodtens - Q;
norm(diff(:)) / norm(Q(:))

% S = C' * X * C;
% Gp = zeros(4,4);
% for i = 1:4
%     for j = 1:4
%         Gp(i,j) = 0;
%         for r1 = 1:4
%             for r2 = 1:4
%                 Gp(i,j) = Gp(i,j) + G(r1,r2,i,j) * S(r1,r2);
%             end
%         end
%     end
% end
% 
% proj1p = tmprod(Gp,{C,C},1:2);
% norm(proj1 - proj1p)
% 
% 
% 
% 
% A = sym('S',[4,4]);
% for i = 1:4
%     for j = 1:4
%         fprintf("(%d,%d)\n",i,j)
%         curval = 0;
%         for r1 = 1:4
%             for r2 = 1:4
%                 curval = curval + G(r1,r2,i,j) * A(r1,r2);
% 
%             end
%         end
%         curval;
%     end
% end
% Gp;

[u,s,v] = svd(proj1);
diag(s)'
rank(proj1)

proj1 = zeros(3*n,3*n);
for i = 1:3*n
    for j = 1:3*n
        proj1 = proj1 + X(i,j)  * reshape(Q(i,:,j,:),[3*n,3*n]);
    end
end

[u,s,v] = svd(proj1);
diag(s)'

proj1 = zeros(3*n,3*n);
for i = 1:3*n
    for j = 1:3*n
        proj1 = proj1 + X(i,j)  * reshape(Q(i,:,:,j),[3*n,3*n]);
    end
end

[u,s,v] = svd(proj1);
diag(s)'

proj1 = zeros(3*n,3*n);
for i = 1:3*n
    for j = 1:3*n
        proj1 = proj1 + X(i,j)  * reshape(Q(:,i,j,:),[3*n,3*n]);
    end
end

[u,s,v] = svd(proj1);
diag(s)'



proj1 = zeros(3*n,3*n);
for i = 1:3*n
    for j = 1:3*n
        proj1 = proj1 + X(i,j)  * reshape(Q(:,i,:,j),[3*n,3*n]);
    end
end

[u,s,v] = svd(proj1);
diag(s)'

proj1 = zeros(3*n,3*n);
for i = 1:3*n
    for j = 1:3*n
        proj1 = proj1 + X(i,j)  * reshape(Q(:,:,i,j),[3*n,3*n]);
    end
end

[u,s,v] = svd(proj1);
diag(s)'

% THE RESULT IS THAT THE PRANK IS ALL 4 WHEN WE PROJECT ONTO MATRICES,
% for all of the cases, the top 2 singular values are equal, and the lower
% 2 singular values are equal. 




% NOW WE CHECK IN THE COLLINEAR CASE, actually there will be no rank drop
% unless we have all cameras being at the same point. This is another
% advantage of quadrifocal tensors. 
[U,V,np,F,t] = generate_random_collinear_cameras(n,true,true);
Q = generate_quad_tensor(np,n);

[U,S,sv] = mlsvd(Q);
sv{1}
sv{2}
sv{3}
sv{4}


T = generate_block_trifocal_tensor(np,n);
[U,S,sv] = mlsvd(T);
sv{1}
sv{2}
sv{3}

% IN COLLINEAR CASE, BLOCK TRIFOCAL TENSOR DROPS RANK TO (4,4,5)
% BLOCK FUNDAMENTAL MATRIX DROPS RANK TO (4)

% rank(F)

% NOW WE CHECK IN THE calibrated case, when we are in the calibrated case,
% all of the singular values are the same. 
% [U,V,np,F,t] = generate_random_projection_cameras(n,false,false);
% Q = generate_quad_tensor(np,n);
% 
% [U,S,sv] = mlsvd(Q);
% sv{1}
% sv{2}
% sv{3}
% sv{4}



