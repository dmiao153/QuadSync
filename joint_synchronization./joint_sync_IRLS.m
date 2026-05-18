function [var, obj_history, res_history] = joint_sync_IRLS(Q,T,E,Wq,Wt,We,C,p,np)

    % here Q and W are both tensors, where Q is the estimated block quadrifocal
    % tensor, W is the sparse tensor of zeros and ones
    % C is the initialization for the camera matrices
    % p is the step size for the ascent step
    MaxIter = 10;
    rel_tol = 10 ^ (-4);

    var = initialize_quad_var(Q,T,E,Wq,Wt,We,C,p);
    cur_obj = quad_obj(var);
    obj_history = zeros(MaxIter+1,1);
    obj_history(1) = cur_obj; 
    res_history = zeros(MaxIter,1);

    diff = inf;
    count = 0;
    while count <= MaxIter && diff > rel_tol
        prev_obj = cur_obj;
        count = count + 1;
        var = calculate_Wq(var);
        var = calculate_Wt(var);
        var = calculate_We(var);

        var = optC(var); % This optimizes the C's, P's and Lambda's iteratively
        var = optB(var);
        var = optD(var);
        var = optG(var);
        var = optT(var);
        
        cur_obj = joint_obj(var);
        obj_history(count + 1) = cur_obj;
        diff = abs((prev_obj - cur_obj) / prev_obj);
        % fprintf("diff: %f \n", diff)
        % fprintf("Iter %d obj: %f\n", count, cur_obj)

        [res] = residual_joint_sync(var, np, 1); % average of c's
        disp('ADMM-avg');
        disp('Relative Cam Difference | Mean Location Error | Median Location Error | Mean Rotation Error | Median Rotation Error |');
        fprintf('%.4f %.4f %.4f %.4f %.4f \n', res.res2, res.errT_mean, res.errT_median, res.errR_mean, res.errR_median);
        res_history(count) = res.res2;
    end

    figure;
    subplot(1,2,1)
    plot(obj_history)
    ylabel('cost function')
    xlabel('Iteration')
    title('Algorithm Cost Progression - joint')

    subplot(1,2,2)
    plot(res_history)
    ylabel('camera residual error')
    xlabel('Iteration')
    title('Residual Progression - joint')

end

function var = calculate_Wq(var)
    G = generate_quad_core();
    for i = 1:var.n
        for j = 1:var.n
            for k = 1:var.n
                for l = 1:var.n
                    if var.Wq(i,j,k,l) > 0
                        var.Wq(i,j,k,l) = 1/max(var.delta,sqrt(tensor_frob(var.Lq(i,j,k,l) * var.Q(3*(i-1)+1:3*i,...
                            3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l) - tmprod(G,{var.c1(3*(i-1)+1:3*i,:), var.c2(3*(j-1)+1:3*j,:),...
                            var.c3(3*(k-1)+1:3*k,:),var.c4(3*(l-1)+1:3*l,:)},1:4))));
                    end
                end
            end
        end
    end         
    % should try to investigate if symmetrizing the weights can help
end

function var = calculate_Wt(var)
    G = generate_core_tensor();
    for i = 1:var.n
        for j = 1:var.n
            for k = 1:var.n
                if var.Wt(i,j,k) > 0
                    var.Wt(i,j,k) = 1/max(var.delta,sqrt(tensor_frob(var.Lt(i,j,k) * var.T(3*(i-1)+1:3*i,...
                        3*(j-1)+1:3*j,3*(k-1)+1:3*k) - tmprod(G,{var.c5(3*(i-1)+1:3*i,:), var.c6(3*(j-1)+1:3*j,:),...
                        var.p1(3*(k-1)+1:3*k,:)},1:3))));
                end
            end
        end
    end         
    % should try to investigate if symmetrizing the weights can help
end

function var = calculate_We(var)
    G = generate_E_core();
    for i = 1:var.n
        for j = 1:var.n
            if var.We(i,j) > 0
                var.We(i,j) = 1/max(var.delta,sqrt(tensor_frob(var.Le(i,j) * var.E(3*(i-1)+1:3*i,...
                    3*(j-1)+1:3*j) - tmprod(G,{var.p2(3*(i-1)+1:3*i,:), var.p3(3*(j-1)+1:3*j,:)},1:2))));
            end
        end
    end         
    % should try to investigate if symmetrizing the weights can help
end


function var = optC(var)
    for i = 1:var.CIter
        var.c1 = sub_optimize_C(var,1);
        var.c2 = sub_optimize_C(var,2);
        var.c3 = sub_optimize_C(var,3);
        var.c4 = sub_optimize_C(var,4);
        var.c5 = sub_optimize_tC(var,1);
        var.c6 = sub_optimize_tC(var,2);
        var.p1 = sub_optimize_tC(var,3);
        var.p2 = sub_optimize_eC(var,1);
        var.p3 = sub_optimize_eC(var,2);
        
        var.Lq = optL(var);
        var.Lt = optLt(var);
        var.Le = optLe(var);
    end
end

function var = optT(var)
    var.tau1 = var.tau1 + (var.p1 - var.D);
    var.tau2 = var.tau2 + (var.p2 - var.D);
    var.tau3 = var.tau3 + (var.p3 - var.D);
end

function var = optG(var)
    var.gamma1 = var.gamma1 + (var.c1 - var.B);
    var.gamma2 = var.gamma2 + (var.c2 - var.B);
    var.gamma3 = var.gamma3 + (var.c3 - var.B);
    var.gamma4 = var.gamma4 + (var.c4 - var.B);
    var.gamma5 = var.gamma5 + (var.c5 - var.B);
    var.gamma6 = var.gamma6 + (var.c6 - var.B);
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

function lambda = optLt(var)
    core = generate_core_tensor();
    nt = tmprod(core,{var.c5,var.c6,var.p1},1:3);
    t = var.T;

    n = var.n;

    lambda = zeros(n,n,n,n);
    if n >= 70
        parfor i = 1:n    
            for j = 1:n
                for k = 1:n
                        if i~=j || j~=k || k~= i
                            cur_ht = tens2mat(t(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                            cur_t = tens2mat(nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                            lambda(i,j,k) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
                    end 
                end
            end
        end
    else
        for i = 1:n    
            for j = 1:n
                for k = 1:n
                    if i~=j || j~=k || k~= i
                        cur_ht = tens2mat(t(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                        cur_t = tens2mat(nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                        lambda(i,j,k) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
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

function lambda = optLe(var)
    core = generate_E_core();
    nt = tmprod(core,{var.p2,var.p3},1:2);
    t = var.E;

    n = var.n;

    lambda = zeros(n,n,n,n);
    if n >= 70
        parfor i = 1:n    
            for j = 1:n
                if i~=j 
                    cur_ht = t(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
                    cur_t = nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
                    lambda(i,j) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
                end 
            end
        end
    else
        for i = 1:n    
            for j = 1:n
                if i~=j 
                    cur_ht = t(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
                    cur_t = nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
                    lambda(i,j,k) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
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

    Ci = zeros(3*var.n,4);
    Wi = tens2mat(repelem(var.Wq,3,3,3,3),i);
    Wi = Wi .^ 2;
    curQ = tens2mat(repelem(var.Lq,3,3,3,3) .* var.Q,i);
    for j = 1:3*var.n
        Ci(j,:) = (Amat(j,:) + ((Wi(j,:).*curQ(j,:)) * K')) ...
            / (var.p * 0.5 * eye(4) + (K .* Wi(j,:))*K');
    end

end

function Ci = sub_optimize_tC(var,i)
    core = generate_core_tensor();
    if i == 1 || i == 2
        Amat = zeros(var.n * 3, 4);
    else
        Amat = zeros(var.n * 3, 6);
    end
    if i == 1
        K = tens2mat(core,1) * kron(var.p1, var.c6)';
        Amat = var.p * 0.5 * (var.B - var.gamma5);
    elseif i ==2
        K = tens2mat(core,2) * kron(var.p1, var.c5)';
        Amat = var.p * 0.5 * (var.B - var.gamma6);
    elseif i == 3
        K = tens2mat(core,3) * kron(var.c6, var.c5)';
        Amat = var.p * 0.5 * (var.D - var.tau1);
    else
        error('optimizing wrong dimension for trifocal tensor opt.')
    end

    if i == 1 || i == 2
        Ci = zeros(3*var.n,4);
        Wi = tens2mat(repelem(var.Wt,3,3,3),i);
        Wi = Wi .^ 2;
        curQ = tens2mat(repelem(var.Lt,3,3,3) .* var.T,i);
        for j = 1:3*var.n
            Ci(j,:) = (Amat(j,:) + ((Wi(j,:).*curQ(j,:)) * K')) ...
                / (var.p * 0.5 * eye(4) + (K .* Wi(j,:)) * K');
        end
    else 
        Ci = zeros(3*var.n,6);
        Wi = tens2mat(repelem(var.Wt,3,3,3),i);
        Wi = Wi .^ 2;
        curQ = tens2mat(repelem(var.Lt,3,3,3) .* var.T,i);
        for j = 1:3*var.n
            Ci(j,:) = (Amat(j,:) + ((Wi(j,:).*curQ(j,:)) * K')) ...
                / (var.p * 0.5 * eye(6) + (K .* Wi(j,:)) * K');
        end
    end
   

end

function Ci = sub_optimize_eC(var,i)
    core = generate_E_core();
    Amat = zeros(var.n * 3, 6);
    if i == 1
        K = tens2mat(core,1) * var.p3';
        Amat = var.p * 0.5 * (var.D - var.p2);
    elseif i == 2 
        K = tens2mat(core,2) * var.p2';
        Amat = var.p * 0.5 * (var.D - var.p3);
    else
        error('optimizing wrong dimension for bifocal tensor opt.')
    end
 
    Ci = zeros(3*var.n,6);
    Wi = tens2mat(repelem(var.We,3,3),i);
    Wi = Wi .^ 2;
    curQ = tens2mat(repelem(var.Le,3,3) .* var.E,i);
    for j = 1:3*var.n
        Ci(j,:) = (Amat(j,:) + ((Wi(j,:).*curQ(j,:)) * K')) ...
            / (var.p * 0.5 * eye(6) + (K .* Wi(j,:)) * K');
    end

end

function var = initialize_quad_var(Q,T,E,Wq,Wt,We,C,p)
    n = size(C,1)/3;
    var.c1 = C;
    var.c2 = C;
    var.c3 = C;
    var.c4 = C;
    var.c5 = C;
    var.c6 = C;

    esC = calculate_stacked_exterior_square_from_mat(C);
    var.p1 = esC;
    var.p2 = esc;
    var.p3 = esc;


    var.n = size(C,1)/3;
    var.B = C;
    var.D = esC;

    var.gamma1 = sparse(3*n,4);
    var.gamma2 = sparse(3*n,4);
    var.gamma3 = sparse(3*n,4);
    var.gamma4 = sparse(3*n,4);
    var.gamma5 = sparse(3*n,4);
    var.gamma6 = sparse(3*n,4);

    var.tau1 = sparse(3*n,6);
    var.tau2 = sparse(3*n,6);
    var.tau3 = sparse(3*n,6);

    var.p = p;
    var.Q = Q;
    var.T = T;
    var.E = E;
    var.Wq = Wq;
    var.Wt = Wt;
    var.We = We;

    var.CIter = 10;

    var.delta = 0.001;

    var.Lq = optL(var);
    var.Lt = optLt(var);
    var.Le = optLe(var);


    var.obj = quad_obj(var);

end

function var = optD(var)

    var.D = (var.p1 + var.p2 + var.p3 + var.tau1 + var.tau2 + var.tau3)/3;

end

function var = optB(var)

    var.B =  (var.c1 + var.c2 + var.c3 + var.c4 + var.c5 + var.c6 + ...
        var.gamma1 + var.gamma2 + var.gamma3 + var.gamma4 + var.gamma5 + var.gamma6)/6;

end

function obj = joint_obj(var)
    quadcore = generate_core();
    tftcore = core_tensor();
    ecore = generate_E_core();

    % quadrifocal tensor cost
    tens1 = repelem(var.Lq,3,3,3,3) .* var.Q;
    tens2 = tmprod(quadcore,{var.c1,var.c2,var.c3,var.c4},1:4);
    quad_obj = tensor_frob(repelem(var.Wq,3,3,3,3) .* (tens1 - tens2));

    % trifocal cost
    tens1 = repelem(var.Lt,3,3,3) .* var.T;
    tens2 = tmprod(tftcore,{var.c5,var.c6,var.p1}, 1:3);
    tens_obj = tensor_frob(repelem(var.Wt,3,3,3) .* (tens1 - tens2));

    % essential matrix cost
    tens1 = repelem(var.Le,3,3) .* var.E;
    tens2 = var.p2 * ecore * transpose(var.p3);
    ess_obj = tensor_frob(repelem(var.We,3,3) .* (tens1-tens2));

    % Creg
    cregobj= 0;
    cregobj = cregobj + norm(var.c1 - var.B + var.gamma1,'fro')^2;
    cregobj = cregobj + norm(var.c2 - var.B + var.gamma2,'fro')^2;
    cregobj = cregobj + norm(var.c3 - var.B + var.gamma3,'fro')^2;
    cregobj = cregobj + norm(var.c4 - var.B + var.gamma4,'fro')^2;
    cregobj = cregobj + norm(var.c5 - var.B + var.gamma5,'fro')^2;
    cregobj = cregobj + norm(var.c6 - var.B + var.gamma6,'fro')^2;
    cregobj = cregobj * var.p * 0.5;

    %Preg
    pregobj = 0;
    pregobj = pregobj + norm(var.p1 - var.D + var.tau1,'fro')^2;
    pregobj = pregobj + norm(var.p2 - var.D + var.tau2,'fro')^2;
    pregobj = pregobj + norm(var.p3 - var.D + var.tau3,'fro')^2;
    pregobj = pregobj * var.p * 0.5;

    fprintf("components of loss %f + %f + %f = %f \n", obj, cregobj, pregobj, obj + cregobj + pregobj);
    obj = obj + cregobj + pregobj;

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

    fprintf("components of loss %f + %f = %f \n", obj, (var.p * obj_reg * 0.5), obj + (var.p * obj_reg * 0.5))
    obj = obj + (var.p * obj_reg * 0.5);

end


function val = tensor_frob(T)
    % this is actually tensor frobenius norm squared
    val = sum((T .^ 2),'all');
end
