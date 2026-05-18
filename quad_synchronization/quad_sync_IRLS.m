function [var, obj_history, res_history] = quad_sync_IRLS(Q,W,C,p, np,t_scale)

% here Q and W are both tensors, where Q is the estimated block quadrifocal
% tensor, W is the sparse tensor of zeros and ones
% C is the initialization for the camera matrices
% p is the step size for the ascent step
MaxIter = 4;
InnerMaxIter = 1;
rel_tol = 10 ^ (-4);

var = initialize_var(Q,W,C,p);
cur_obj = quad_obj(var);
obj_history = zeros(MaxIter+1,1);
obj_history(1) = cur_obj; 
res_history = zeros(MaxIter,1);

diff = inf;
count = 0;
[res] = residual_quad_sync(var, np, t_scale); % average of c's
disp('Iter | Cost | Relative Cam Difference | Mean Location Error | Median Location Error | Mean Rotation Error | Median Rotation Error |');
var = calculate_W(var);

%disp('ADMM-avg');
%disp('Relative Cam Difference | Mean Location Error | Median Location Error | Mean Rotation Error | Median Rotation Error |');
fprintf('%d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, \n', count, cur_obj, res.res2, res.errT_mean, res.errT_median, res.errR_mean, res.errR_median);
while count <= MaxIter-1 && diff > rel_tol
    prev_obj = cur_obj;
    count = count + 1;
    innercount = 0;
    while innercount < InnerMaxIter
        var = optC(var);
        var = optB(var);
        var = optG(var);

        var = adjust_penalty(var);
        
        innercount = innercount + 1;
        

        cur_obj = quad_obj(var);
        obj_history(count + 1) = cur_obj;
        diff = abs((prev_obj - cur_obj) / prev_obj);
        % fprintf("diff: %f \n", diff)
        % fprintf("Iter %d obj: %f\n", count, cur_obj)
    
        [res] = residual_quad_sync(var, np, t_scale); % average of c's
        %disp('ADMM-avg');
        %disp('Relative Cam Difference | Mean Location Error | Median Location Error | Mean Rotation Error | Median Rotation Error |');
        fprintf('%d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, \n', count, cur_obj, res.res2, res.errT_mean, res.errT_median, res.errR_mean, res.errR_median);
    end
    
    var = calculate_W(var);

    res_history(count) = res.res2;
end

figure;
subplot(1,2,1)
plot(obj_history)
ylabel('cost function')
xlabel('Iteration')
title('Algorithm Cost Progression')

subplot(1,2,2)
plot(res_history)
ylabel('camera residual error')
xlabel('Iteration')
title('Residual Progression')

end

function var = calculate_W(var)
G = generate_quad_core();
for i = 1:var.n
    for j = 1:var.n
        for k = 1:var.n
            for l = 1:var.n
                if var.W(i,j,k,l) > 0
                    var.W(i,j,k,l) = 1/max(var.delta,sqrt(tensor_frob(var.L(i,j,k,l) * var.Q(3*(i-1)+1:3*i,...
                        3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l) - tmprod(G,{var.c1(3*(i-1)+1:3*i,:), var.c2(3*(j-1)+1:3*j,:),...
                        var.c3(3*(k-1)+1:3*k,:),var.c4(3*(l-1)+1:3*l,:)},1:4))));
                end
            end
        end
    end
end

              
% should try to investigate if symmetrizing the weights can help
var.W = symmetrize(var.W);
end


function var = optC(var)

for i = 1:var.CIter
var.c1 = sub_optimize_C(var,1);
var.c2 = sub_optimize_C(var,2);
var.c3 = sub_optimize_C(var,3);
var.c4 = sub_optimize_C(var,4);
var.L = optL(var);
end

end

function var = optG(var)
var.gamma1 = var.gamma1 + (var.c1 - var.B);
var.gamma2 = var.gamma2 + (var.c2 - var.B);
var.gamma3 = var.gamma3 + (var.c3 - var.B);
var.gamma4 = var.gamma4 + (var.c4 - var.B);
% var.gamma1 = var.gamma1 + var.p * (var.c1 - var.B);
% var.gamma2 = var.gamma2 + var.p * (var.c2 - var.B);
% var.gamma3 = var.gamma3 + var.p * (var.c3 - var.B);
% var.gamma4 = var.gamma4 + var.p * (var.c4 - var.B);
end

function lambda = optL(var)
core = generate_quad_core();
nt = tmprod(core,{var.c1,var.c2,var.c3,var.c4},1:4);
t = var.Q;

n = var.n;

lambda = zeros(n,n,n,n);
if n >= 70
    parfor i = 1:n    
        for j = 1:n
            for k = 1:n
                for l = 1:n
                    if i~=j || j~=k || k~= l || i~=l
                        cur_ht = tens2mat(t(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k, 3*(l-1)+1:3*l),1);
                        cur_t = tens2mat(nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l),1);
                        lambda(i,j,k,l) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
                    end
                end 
            end
        end
    end
else
    for i = 1:n    
        for j = 1:n
            for k = 1:n
                for l = 1:n
                    if i~=j || j~=k || k~= l || i~=l
                        cur_ht = tens2mat(t(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k, 3*(l-1)+1:3*l),1);
                        cur_t = tens2mat(nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l),1);
                        lambda(i,j,k,l) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
                    end
                end
            end
        end
    end
end

lambda(isnan(lambda)) = 0;
lambda = symmetrize(lambda);
lambda = lambda / norm(lambda(:));
% fprintf("Norm of L is %d \n", norm(lambda(:)))
end

function Ci = sub_optimize_C(var,i)
core = generate_quad_core();
Amat = zeros(var.n * 3, 4);
if i == 1
    K = tens2mat(core,1) * kron(kron(var.c4, var.c3),var.c2)';
    Amat = var.p * 0.5 * (var.B - var.gamma1);
elseif i ==2
    K = tens2mat(core,2) * kron(kron(var.c4, var.c3),var.c1)';
    Amat = var.p * 0.5 * (var.B - var.gamma2);
elseif i ==3
    K = tens2mat(core,3) * kron(kron(var.c4, var.c2),var.c1)';
    Amat = var.p * 0.5 * (var.B - var.gamma3);
elseif i ==4
    K = tens2mat(core,4) * kron(kron(var.c3, var.c2),var.c1)';
    Amat = var.p * 0.5 * (var.B - var.gamma4);
else
    error('optimizing wrong dimension for quadrifocal tensor opt.')
end
% if i == 1
%     K = tens2mat(core,1) * kron(kron(var.c2, var.c3),var.c4)';
%     Amat = var.p * 0.5 * (var.B - var.gamma1);
% elseif i ==2
%     K = tens2mat(core,2) * kron(kron(var.c1, var.c3),var.c4)';
%     Amat = var.p * 0.5 * (var.B - var.gamma2);
% elseif i ==3
%     K = tens2mat(core,3) * kron(kron(var.c1, var.c2),var.c4)';
%     Amat = var.p * 0.5 * (var.B - var.gamma3);
% elseif i ==4
%     K = tens2mat(core,4) * kron(kron(var.c1, var.c2),var.c3)';
%     Amat = var.p * 0.5 * (var.B - var.gamma4);
% else
%     error('optimizing wrong dimension for quadrifocal tensor opt.')
% end

Ci = zeros(3*var.n,4);
Wi = tens2mat(repelem(var.W,3,3,3,3),i);
Wi = Wi .^ 2;
curQ = tens2mat(repelem(var.L,3,3,3,3) .* var.Q,i);
for j = 1:3*var.n
    Ci(j,:) = (Amat(j,:) + ((Wi(j,:).*curQ(j,:)) * K')) ...
        / (var.p * 0.5 * eye(4) + (K .* Wi(j,:))*K');
end

end

function var = initialize_var(Q,W,C,p)
n = size(C,1)/3;
var.c1 = C;
var.c2 = C;
var.c3 = C;
var.c4 = C;

var.n = size(C,1)/3;
var.B = C;
var.prevB = C;
var.gamma1 = sparse(3*n,4);
var.gamma2 = sparse(3*n,4);
var.gamma3 = sparse(3*n,4);
var.gamma4 = sparse(3*n,4);
var.p = p;
var.Q = Q;
var.W = W;
var.CIter = 10;

var.delta = 0.001;

var.L = optL(var);

var.obj = quad_obj(var);

end

function var = optB(var)
var.prevB = var.B;
var.B = 0.25 * (var.c1 + var.c2 + var.c3 + var.c4 + var.gamma1 + ...
    var.gamma2 + var.gamma3 + var.gamma4);

end

function obj = quad_obj(var)
core = generate_core();

obj = 0;

tens1 = repelem(var.L,3,3,3,3) .* var.Q;
tens2 = tmprod(core,{var.c1,var.c2,var.c3,var.c4},1:4);
obj = obj + tensor_frob(repelem(var.W,3,3,3,3) .* (tens1 - tens2));

obj_reg = 0;
obj_reg = obj_reg + norm(var.c1 - var.B + var.gamma1,'fro')^2;
obj_reg = obj_reg + norm(var.c2 - var.B + var.gamma2,'fro')^2;
obj_reg = obj_reg + norm(var.c3 - var.B + var.gamma3,'fro')^2;
obj_reg = obj_reg + norm(var.c4 - var.B + var.gamma4,'fro')^2;

% fprintf("components of loss %f + %f = %f \n", obj, (var.p * obj_reg * 0.5), obj + (var.p * obj_reg * 0.5))
obj = obj + (var.p * obj_reg * 0.5);

end


function val = tensor_frob(T)
% this is actually tensor frobenius norm squared
val = sum((T .^ 2),'all');
end

function var = adjust_penalty(var)
mu = 10;
tauinc = 2;
taudec = 2;

gammaknorm = norm(var.c1 - var.B) + norm(var.c2 - var.B) + norm(var.c3 - var.B) + norm(var.c4 - var.B);
sknorm = 4 * var.p * norm(var.B - var.prevB);
if gammaknorm > mu * sknorm
    var.p = var.p*tauinc;
    var.gamma1 = var.gamma1 / tauinc;
    var.gamma2 = var.gamma2 / tauinc;
    var.gamma3 = var.gamma3 / tauinc;
    var.gamma4 = var.gamma4 / tauinc;
end
if sknorm > mu * gammaknorm
    var.p = var.p / taudec;
    var.gamma1 = var.gamma1 * taudec;
    var.gamma2 = var.gamma2 * taudec;
    var.gamma3 = var.gamma3 * taudec;
    var.gamma4 = var.gamma4 * taudec;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
