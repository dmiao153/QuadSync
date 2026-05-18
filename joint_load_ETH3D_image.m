% This function jointly optimizes the block quadrifocal tensor, the block trifocal tensor, and the block essential matrix
% Using an IRLS aproach. 
% clear all;
function joint_load_ETH3D_image(dataset)
    s = rng;
    rng(1)

    dataset = "relief";
    diary nov1nightrun.txt

    load("Data_ETH3D/" + dataset + "/" + dataset + "_data368.mat")
    n = data2.n;
    t_scale = 1;

    % retrieve the triplet points by taking the intersection of points between pair views
    corresp_by_triplet = get_points_triplets(data2);
    ccams = generate_gt_cams(data2);

    % initialize bundle-adjustment parameters
    initial_sample_size=100;
    bundle_adj_size=50;
    repr_err_th=1;

    methods={@STETFTPoseEstimation};     

    bad_reprojection = [];
    methods_to_test = 1;

    %% error vectors
    repr_err = zeros(n^3,length(methods),2);
    rot_err  = zeros(n^3,length(methods),2);
    t_err    = zeros(n^3,length(methods),2);
    iter     = zeros(n^3,length(methods),2);
    time     = zeros(n^3,length(methods),2);
    time_estimate = zeros(n^3,4);

    %% evaluation
    cTest = cell(n,n,n);
    tftkeep = sparse(zeros((n*(n-1))/2-1,n));

    it = 1;
    tic
    for i = 1:n
        for j = 1:n
            for k = 1:n
                if i < j && j < k && (size(data2.E_est{i,j},1) > 0) && ...
                        (size(data2.E_est{i,k},1) > 0) && (size(data2.E_est{j,k},1) > 0)

                    % Triplet information and correspondances

                    im1 = i; im2 = j; im3 = k; 
                    Corresp=corresp_by_triplet{im1,im2,im3}';

                    if size(Corresp,2) < 10
                        continue
                    end

                    N=size(Corresp,2);
                    fprintf('Triplet %d/%d (%d,%d,%d) with %d matching points.\n',...
                        it,n^3,im1,im2,im3,N);

                    % Ground truth poses and calibration

    %                [K1,R1_true,t1_true,im_size]=readCalibrationOrientation_EPFL(path_to_data,im_names{im1});
    %                [K2,R2_true,t2_true]=readCalibrationOrientation_EPFL(path_to_data,im_names{im2});
    %                [K3,R3_true,t3_true]=readCalibrationOrientation_EPFL(path_to_data,im_names{im3});
                    CalM=[data2.K(:,:,i);data2.K(:,:,j);data2.K(:,:,k)];
    %                R_t0={[R2_true*R1_true.', t2_true-R2_true*R1_true.'*t1_true],...
    %                    [R3_true*R1_true.', t3_true-R3_true*R1_true.'*t1_true]};
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    Corresp_inliers = Corresp;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    % if size(Corresp,2) > 1500
                    %     rng(it);
                    %     init_sample=randsample(1:N,1500);
                    %     rng(it);
                    %     Corresp_init=Corresp_inliers(:,init_sample);
                    %     Corresp = Corresp_init;
                    % end

                    % estimate the pose using all methods
                    fprintf('method... ');
                    for m=methods_to_test
                        fprintf('%d ',m);

                        % if there are not enough matches for initial estimation, inf error
                        if N<10 
                            repr_err(it,m,:)=inf;    rot_err(it,m,:)=inf;
                            t_err(it,m,:)=inf;       iter(it,m,:)=inf;
                            time(it,m,:)=inf;
                            continue;
                        end

                        % % STE POSE ESTIMATION      TIMING 1
                        t0=cputime;
                        [R_t_2,R_t_3,Reconst,T,nit, distances]=methods{m}(Corresp,CalM);   % STE to find 
                        t=cputime-t0;
                        time_estimate(it, 1) = t;

                        t0 = cputime;
                        [T, distances2, Corresp_inliers, R_t_2, R_t_3] = STE_local_optimization(T, Corresp', distances, 0.4, CalM);
                        t=cputime-t0;
                        time_estimate(it, 2) = t;
                        % % STE POSE ESTIMATION       

                        Corresp_inliers = Corresp_inliers';
                        N = size(Corresp_inliers,2);

                        rng(it);
                        init_sample=randsample(1:N,min(initial_sample_size,N));
                        rng(it);
                        ref_sample=randsample(init_sample,min(bundle_adj_size,length(init_sample)));
                        Corresp_inliers=Corresp_inliers(:,init_sample);
                        Corresp_ref=Corresp_inliers(:,ref_sample);


                        % % Apply Bundle Adjustment TIMING 2
                        fprintf('(ref)... ');
                        t0=cputime;
                        [R_t_ref,~,nit,repr_errBA]=BundleAdjustment(CalM,...
                            [eye(3,4);R_t_2;R_t_3],Corresp_ref);
                        t=cputime-t0;
                        time_estimate(it, 3) = t;

                        % reprojection error with all inliers TIMING 3
                        t0 = cputime;
                        repr_err(it,m,2)= ReprError({CalM(1:3,:)*R_t_ref(1:3,:),...
                            CalM(4:6,:)*R_t_ref(4:6,:),...
                            CalM(7:9,:)*R_t_ref(7:9,:)},Corresp_inliers);

                        ref_R_t = [R_t_ref(1:3,:);R_t_ref(4:6,:);R_t_ref(7:9,:)];
                        %ref_R_t = transform_R_t_ref(ref_R_t);

                        if repr_err(it,m,2) < 1
                            %%%%%%%%%%%%%%%%%%%%
                            rowindex = sum(n-1:-1:n-i+1) + (j - i);
                            colindex = k;
                            fprintf("Row: %d, Col: %d\n", rowindex, colindex);
                            tftkeep(rowindex,colindex) = 1;
                            cTest{i,j,k} = ref_R_t;
                            %%%%%%%%%%%%%%%%%%%%
                            % cTest{im2,im3,im1} = T_from_P({R_t_ref(1:3,:),R_t_ref(4:6,:), R_t_ref(7:9,:)});
                            % cTest{im3,im2,im1} = T_from_P({R_t_ref(1:3,:),R_t_ref(7:9,:), R_t_ref(4:6,:)});
                            % cTest{im1,im3,im2} = T_from_P({R_t_ref(4:6,:),R_t_ref(1:3,:), R_t_ref(7:9,:)});
                            % cTest{im3,im1,im2} = T_from_P({R_t_ref(4:6,:),R_t_ref(7:9,:), R_t_ref(1:3,:)});
                            % cTest{im1,im2,im3} = T_from_P({R_t_ref(7:9,:),R_t_ref(1:3,:), R_t_ref(4:6,:)});
                            % cTest{im2,im1,im3} = T_from_P({R_t_ref(7:9,:),R_t_ref(4:6,:), R_t_ref(1:3,:)});
                        else
                            bad_reprojection = [bad_reprojection; i,j,k];
                        end
                        t = cputime - t0;
                        time_estimate(it, 4) = t;

                    end
                    it = it + 1; 
                end
            end
        end
    end
    toc

    % load("EPFL_GlueStick/"+dataset+"_two_view_data.mat");
    % for i = 1:n
    %     for j = 1:n
    %         if i ~= j
    %             cTest{i,i,j} = reorder_iij_from_fundamental(data.E_est{i,j});
    %         end
    %     end
    % end

    load("./2_quadrifocal_tensor_synchronization/"+dataset +"tft_check_dataset.mat")

    cameras_gt = cell(1,n);
    for i = 1:n
        cameras_gt{i} = ccams(:,:,i); 
    end


    cycles = find_cycles(tftkeep);
    good_cycles = ones(size(cycles,1),1);

    check_positions = zeros(n,n,n,n);

    rot_inconsistency = zeros(size(cycles,1),1);
    trans_inconsistency = zeros(size(cycles,1),1);
    quadrifocal_tensor = zeros(3*n,3*n,3*n,3*n);
    for t = 1:size(cycles,1)
        cycle_cams = zeros(9,4,4);
        if size(cTest{cycles(t,1),cycles(t,2),cycles(t,3)},1) > 0 && ...
                size(cTest{cycles(t,2),cycles(t,3),cycles(t,4)},1) > 0 && ...
                size(cTest{cycles(t,1),cycles(t,3),cycles(t,4)},1) > 0 && ...
                size(cTest{cycles(t,1),cycles(t,2),cycles(t,4)},1) > 0 
            
            cycle_cams(:,:,1) = cTest{cycles(t,1),cycles(t,2),cycles(t,3)};
            cycle_cams(:,:,2) = cTest{cycles(t,2),cycles(t,3),cycles(t,4)};
            cycle_cams(:,:,3) = cTest{cycles(t,1),cycles(t,3),cycles(t,4)};
            cycle_cams(:,:,4) = cTest{cycles(t,1),cycles(t,2),cycles(t,4)};


            cur_cycle_cams = zeros(9,4,4);
            cur_cycle_cams(:,:,1) = cycle_cams(:,:,1);
            cur_cycle_cams(:,:,2) = cycle_cams(:,:,2);
            cur_cycle_cams(:,:,3) = [cycle_cams(4:6,:,3);cycle_cams(7:9,:,3);cycle_cams(1:3,:,3)];
            cur_cycle_cams(:,:,4) = [cycle_cams(7:9,:,4);cycle_cams(1:3,:,4);cycle_cams(4:6,:,4)];
            [rot_err, transerr] = consistency_measure(cur_cycle_cams);
            if rot_err > 3 || transerr > 0.2
                good_cycles(t) = 0;
                continue;
            end

            [cam1est, cam2est, cam3est, cam4est] = calculate_initial_quad_cameras(cycle_cams);
            
            i = cycles(t,1);
            j = cycles(t,2);
            k = cycles(t,3);
            l = cycles(t,4);

            quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l) = Quad_from_P(cam1est,cam2est,cam3est,cam4est);
            quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(l-1)+1:3*l,3*(k-1)+1:3*k) = Quad_from_P(cam1est,cam2est,cam4est,cam3est);
            quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(l-1)+1:3*l) = Quad_from_P(cam1est,cam3est,cam2est,cam4est);
            quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(l-1)+1:3*l,3*(j-1)+1:3*j) = Quad_from_P(cam1est,cam3est,cam4est,cam2est);
            quadrifocal_tensor(3*(i-1)+1:3*i,3*(l-1)+1:3*l,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(cam1est,cam4est,cam2est,cam3est);
            quadrifocal_tensor(3*(i-1)+1:3*i,3*(l-1)+1:3*l,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(cam1est,cam4est,cam3est,cam2est);


            quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(l-1)+1:3*l) = Quad_from_P(cam2est,cam1est,cam3est,cam4est);
            quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(l-1)+1:3*l,3*(k-1)+1:3*k) = Quad_from_P(cam2est,cam1est,cam4est,cam3est);
            quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(l-1)+1:3*l) = Quad_from_P(cam2est,cam3est,cam1est,cam4est);
            quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l,3*(i-1)+1:3*i) = Quad_from_P(cam2est,cam3est,cam4est,cam1est);
            quadrifocal_tensor(3*(j-1)+1:3*j,3*(l-1)+1:3*l,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(cam2est,cam4est,cam1est,cam3est);
            quadrifocal_tensor(3*(j-1)+1:3*j,3*(l-1)+1:3*l,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(cam2est,cam4est,cam3est,cam1est);


            quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(l-1)+1:3*l) = Quad_from_P(cam3est,cam1est,cam2est,cam4est);
            quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(l-1)+1:3*l,3*(j-1)+1:3*j) = Quad_from_P(cam3est,cam1est,cam4est,cam2est);
            quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(l-1)+1:3*l) = Quad_from_P(cam3est,cam2est,cam1est,cam4est);
            quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(l-1)+1:3*l,3*(i-1)+1:3*i) = Quad_from_P(cam3est,cam2est,cam4est,cam1est);
            quadrifocal_tensor(3*(k-1)+1:3*k,3*(l-1)+1:3*l,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(cam3est,cam4est,cam1est,cam2est);
            quadrifocal_tensor(3*(k-1)+1:3*k,3*(l-1)+1:3*l,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(cam3est,cam4est,cam2est,cam1est);


            quadrifocal_tensor(3*(l-1)+1:3*l,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(cam4est,cam1est,cam2est,cam3est);
            quadrifocal_tensor(3*(l-1)+1:3*l,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(cam4est,cam1est,cam3est,cam2est);
            quadrifocal_tensor(3*(l-1)+1:3*l,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(cam4est,cam2est,cam1est,cam3est);
            quadrifocal_tensor(3*(l-1)+1:3*l,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(cam4est,cam2est,cam3est,cam1est);
            quadrifocal_tensor(3*(l-1)+1:3*l,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(cam4est,cam3est,cam1est,cam2est);
            quadrifocal_tensor(3*(l-1)+1:3*l,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(cam4est,cam3est,cam2est,cam1est);

            check_positions(i,j,k,l) = 1;

        end

    end

    %%% We will add in the fundamental and trifocal tensor information into the quadrifocal tensor
    % Trifocal tensors first
    trifocal_tensor = zeros(3*n,3*n,3*n);

    for i = 1:n 
        for j = 1:n 
            for k = 1:n 
                if i < j && j < k && size(cTest{i,j,k},1) == 9
                    camest1 = cTest{i,j,k}(1:3,:);
                    camest2 = cTest{i,j,k}(4:6,:);
                    camest3 = cTest{i,j,k}(7:9,:);

                    %cam1 repeated
                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(camest1,camest1,camest2,camest3);
                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(camest1,camest1,camest3,camest2);

                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(camest1,camest2,camest1,camest3);
                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(camest1,camest2,camest3,camest1);

                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(camest1,camest3,camest1,camest2);
                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(camest1,camest3,camest2,camest1);

                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(camest2,camest1,camest1,camest3);
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(camest2,camest1,camest3,camest1);

                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(camest3,camest1,camest1,camest2);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(camest3,camest1,camest2,camest1);

                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(i-1)+1:3*i) = Quad_from_P(camest2,camest3,camest1,camest1);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(i-1)+1:3*i) = Quad_from_P(camest3,camest2,camest1,camest1);


                    %cam2 repeated
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(camest2,camest2,camest1,camest3);
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(camest2,camest2,camest3,camest1);

                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(camest2,camest1,camest2,camest3);
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(camest2,camest1,camest3,camest2);

                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(camest2,camest3,camest2,camest1);
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(camest2,camest3,camest1,camest2);

                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(camest1,camest2,camest2,camest3);
                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(camest1,camest2,camest3,camest2);

                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(camest3,camest2,camest2,camest1);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(camest3,camest2,camest1,camest2);

                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(j-1)+1:3*j) = Quad_from_P(camest1,camest3,camest2,camest2);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(j-1)+1:3*j) = Quad_from_P(camest3,camest1,camest2,camest2);


                    %cam3 repeated
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = Quad_from_P(camest3,camest3,camest1,camest2);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = Quad_from_P(camest3,camest3,camest2,camest1);

                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(camest3,camest1,camest3,camest2);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(camest3,camest1,camest2,camest3);

                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(camest3,camest2,camest3,camest1);
                    quadrifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(camest3,camest2,camest1,camest3);

                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = Quad_from_P(camest1,camest3,camest3,camest2);
                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = Quad_from_P(camest1,camest3,camest2,camest3);

                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = Quad_from_P(camest2,camest3,camest3,camest1);
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = Quad_from_P(camest2,camest3,camest1,camest3);

                    quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(k-1)+1:3*k) = Quad_from_P(camest1,camest2,camest3,camest3);
                    quadrifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(k-1)+1:3*k) = Quad_from_P(camest2,camest1,camest3,camest3);



                    trifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k) = T_from_P({camest3,camest1,camest2});
                    trifocal_tensor(3*(i-1)+1:3*i,3*(k-1)+1:3*k,3*(j-1)+1:3*j) = T_from_P({camest2,camest1,camest3});
                    trifocal_tensor(3*(j-1)+1:3*j,3*(i-1)+1:3*i,3*(k-1)+1:3*k) = T_from_P({camest3,camest2,camest1});
                    trifocal_tensor(3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(i-1)+1:3*i) = T_from_P({camest1,camest2,camest3});
                    trifocal_tensor(3*(k-1)+1:3*k,3*(i-1)+1:3*i,3*(j-1)+1:3*j) = T_from_P({camest3,camest3,camest1});
                    trifocal_tensor(3*(k-1)+1:3*k,3*(j-1)+1:3*j,3*(i-1)+1:3*i) = T_from_P({camest1,camest3,camest2});

                end
            end
        end
    end

    essential_matrix = E_est_cell_to_array(data2.E_est);


    %% Now let us try a smaller pruned dataset and see if the results improve

    % It is better to use some tensor multiway clustering algorithm here 
    densities = check_ind_density(cycles,n);
    cutoff_mask = sort(find(densities > 0.05));
    cutoff_mask = screen_cutoff_mask(cutoff_mask, check_positions);

    if length(cutoff_mask) <= 5
        cutoff_mask = sort(find(densities > 0.02));
        cutoff_mask = screen_cutoff_mask(cutoff_mask, check_positions);
    end
    if length(cutoff_mask) <= 5
        cutoff_mask = sort(find(densities > 0.01));
        cutoff_mask = screen_cutoff_mask(cutoff_mask, check_positions);
    end


    nn = length(cutoff_mask);
    fprintf("The final pruned dataset size is %d/%d \n", nn, n)
    cameras_gt = cameras_gt(cutoff_mask);
    final_indices = sort([3*(cutoff_mask-1)+1; 3*(cutoff_mask-1)+2; 3*(cutoff_mask-1)+3]);
    quadrifocal_tensor = quadrifocal_tensor(final_indices,final_indices,final_indices,final_indices);
    trifocal_tensor = trifocal_tensor(final_indices,final_indices,final_indices);
    essential_matrix = essential_matrix(final_indices,final_indices);

    completion_rate = check_completion_rate(quadrifocal_tensor, nn);
    fprintf("The Completion Rate is %.4f \n", completion_rate)
    %%================================================================================================================
    %% Parameter Search for IRLS quadrifocal synchronization
    nqt = normalize_quadrifocal_tensor(quadrifocal_tensor);
    [Wq,C]=initialization_for_quadsync(nqt);
    trifocal_tensor = update_scales_onT(C,trifocal_tensor);
    [Wt]=initialization_for_tftsync(trifocal_tensor);
    essential_matrix = update_scales_onE(C,essential_matrix);
    [We]=initialization_for_esync(essential_matrix);

    %pss = [10,1,0.1,0.01,0.001,0.0001];
    pss = [0.00001];
    for j = 1:length(pss)
        curpss = pss(j);

        tic
            [var, obj_history, res_history] = joint_sync_tensors_IRLS(nqt,trifocal_tensor,essential_matrix,Wq,Wt,We,C,curpss,cameras_gt);
        toc 
        
        [res] = residual_quad_sync(var, cameras_gt, 1); % average of c's
        disp('ADMM-IRLS');
        fprintf("The current p is %f \n", curpss)
        disp('Relative Cam Difference | Mean Location Error | Median Location Error | Mean Rotation Error | Median Rotation Error |');
        fprintf('%.4f %.4f %.4f %.4f %.4f \n', res.res2, res.errT_mean, res.errT_median, res.errR_mean, res.errR_median);
        fprintf('-quadIRLS ADMM Rel Cam Diff: %f \n', res.res2);

    end


    % diary off
end




function completion_rate = check_completion_rate(quadrifocal_tensor, n)

    total_positions = n^4;
    counter = 0;
    for i =1:n 
        for j= 1:n 
            for k = 1:n 
                for l = 1:n 
                    if nnz(quadrifocal_tensor(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k,3*(l-1)+1:3*l)) > 0
                        counter = counter + 1;
                    end
                end
            end
        end
    end

    completion_rate = (counter + n) / total_positions;

end


function new_cutoff_mask = screen_cutoff_mask(cutoff_mask, check_positions)
    % just avoid that there are zero measurements. 
    new_check_positions = check_positions(cutoff_mask, cutoff_mask, cutoff_mask, cutoff_mask);
    new_cutoff_mask = [];
    for i = 1 : length(cutoff_mask)
        if sum(new_check_positions(i,:,:,:),'all') > 0 ||  sum(new_check_positions(:,i,:,:),'all') > 0 || ...
            sum(new_check_positions(:,:,i,:),'all') > 0 || sum(new_check_positions(:,:,:,i),'all') > 0
            new_cutoff_mask = [new_cutoff_mask, cutoff_mask(i)];
        end
    end

end



function Earr = E_est_cell_to_array(E_est)
    n = size(E_est,1);
    Earr = zeros(3*n,3*n);
    for i = 1:n
        for j =1:n
            if size(E_est{i,j},1)>0
                Earr(3*(i-1)+1:3*i,3*(j-1)+1:3*j) = E_est{i,j};
            end
        end
    end

end

function essmat = update_scales_onE(C,essential_mat)
    n = size(C,1)/3;
    estimated_tensor = generate_block_fundamental_matrix(reorder_np_array_to_cell(C),n);
    lambda = lambda_from_admm_fore(estimated_tensor, essential_mat);
    essmat = essential_mat .* repelem(lambda,3,3);
    essmat(isnan(essmat)) = 0;
end

function trifocal_tensor = update_scales_onT(C,trifocal_tensor)
    n = size(C,1)/3;
    estimated_tensor = generate_block_trifocal_tensor(reorder_np_array_to_cell(C),n);
    lambda = lambda_from_admm(estimated_tensor, trifocal_tensor);
    trifocal_tensor = trifocal_tensor .* repelem(lambda,3,3,3);
    trifocal_tensor(isnan(trifocal_tensor)) = 0;
end

function densities = check_ind_density(cycles,n)
    base_total = (n-1) * (n-2) * (n-3) / 6;
    densities = zeros(n,1);
    
    for i = 1:n
        densities(i) = sum(cycles == i,'all') / base_total;
    end

end

function triplet_points = get_points_triplets(data)
    triplet_points = cell(data.n,data.n,data.n);
    for i = 1:data.n 
        for j = 1:data.n 
            for k = 1:data.n 
                if i < j && j < k && size(data.corr{i,j},1) > 0 && size(data.corr{i,k},1)>0 && size(data.corr{j,k},1)>0
                    [~, ia, ib] = intersect(transpose(data.corr{i,j}(:,:,2)), transpose(data.corr{j,k}(:,:,1)), 'rows');
                    if isempty(ia)
                        continue;
                    end
                    corrijk = zeros(2,length(ia),3);
                    corrijk(:,:,1:2) = data.corr{i,j}(:,ia,:);
                    corrijk(:,:,3) = data.corr{j,k}(:,ib,2);
                    
                    [~, nia, ~] = intersect(transpose([corrijk(:,:,1);corrijk(:,:,3)]), ...
                        transpose([data.corr{i,k}(:,:,1);data.corr{i,k}(:,:,2)]), 'rows');
                    if isempty(nia)
                        continue;
                    end
                    triplet_points{i,j,k} = transpose([corrijk(:,nia,1);corrijk(:,nia,2);corrijk(:,nia,3)]);
                end
            end
        end
    end

end

function ccams = generate_gt_cams(data2)

    ccams = zeros(3,4,data2.n);
    for i = 1:data2.n
        ccams(:,:,i) = [data2.R(:,:,i), data2.t(:,:,i)];

        % ccams(:,:,i) = [data2.R(:,:,i), data2.R(:,:,i) * data2.t(:,:,i)];
    end


end

function cycles = find_cycles(keep)
    % keep is a [(n choose 2)-1] x n sparse matrix that encodes whether a trifocal tensor is measured or not
    n = size(keep,2);
    block_indices = cumsum(n-1:-1:1);
    block_indices = [0,block_indices,block_indices(end)+1];
    cycles = [];
    % the rows are ordered as 12,13,14,...,1n,23,24,....,2n,34,35,...,3n,....
    % lengths of each block would be n,n-1,...,1. So for example of n = 5, we will have [0,5,9,12,14,15,16]
    for i = 1:size(keep,1)
        curind = find(i<=block_indices,1,'first');
        ind1 = curind -1;
        ind2 = i + ind1 - block_indices(ind1);
        % fprintf("(%d,%d)\n",ind1,ind2);
        observed_indices = find(keep(i,:));
        if length(observed_indices) > 1
            cur_subinds = nchoosek(observed_indices,2);
            new_cycles = [kron(ind1,ones(size(cur_subinds,1),1)),...
                    kron(ind2,ones(size(cur_subinds,1),1)),cur_subinds];
            cycles = [cycles;new_cycles];
        end
    end
end

function [cam1est, cam2est, cam3est, cam4est] = calculate_initial_quad_cameras(cycles_cams)
    % cycle_cams is an sptensor of size 9 x 4 x 4, so that the three cameras are concatenated on top of each other
    % and (:,:,i) are the cameras for the ith index in the cycle
    % 123,234,341,412

    %123,234,134,124
    
    [Hf1, ~] = findHomography({cycles_cams(1:3,:,2),cycles_cams(4:6,:,2)}, ...
            {cycles_cams(4:6,:,1),cycles_cams(7:9,:,1)});
    
    cycles_cams(:,:,2) = cycles_cams(:,:,2) * Hf1;
    
    [Hf2, ~] = findHomography({cycles_cams(4:6,:,3),cycles_cams(7:9,:,3)}, ...
            {cycles_cams(4:6,:,2),cycles_cams(7:9,:,2)});
    
    cycles_cams(:,:,3) = cycles_cams(:,:,3) * Hf2;
    
    [Hf3, ~] = findHomography({cycles_cams(1:3,:,4),cycles_cams(7:9,:,4)}, ...
            {cycles_cams(1:3,:,3),cycles_cams(7:9,:,3)});
    
    cycles_cams(:,:,4) = cycles_cams(:,:,4) * Hf3;
    
    for i = 1:4
        cycles_cams(1:3,:,i) = cycles_cams(1:3,:,i) / norm(reshape(cycles_cams(1:3,:,i),[3,4]),'fro');
        cycles_cams(4:6,:,i) = cycles_cams(4:6,:,i) / norm(reshape(cycles_cams(4:6,:,i),[3,4]),'fro');
        cycles_cams(7:9,:,i) = cycles_cams(7:9,:,i) / norm(reshape(cycles_cams(7:9,:,i),[3,4]),'fro');
    end

    cam1est = cycles_cams(1:3,:,1) + cycles_cams(1:3,:,3) + cycles_cams(1:3,:,4);
    cam2est = cycles_cams(4:6,:,1) + cycles_cams(1:3,:,2) + cycles_cams(4:6,:,4);
    cam3est = cycles_cams(7:9,:,1) + cycles_cams(4:6,:,2) + cycles_cams(4:6,:,3);
    cam4est = cycles_cams(7:9,:,2) + cycles_cams(7:9,:,3) + cycles_cams(7:9,:,4);    

end

function lambda = lambda_from_admm_fore(nt,t)
n = size(nt,1)/3;

lambda = zeros(n,n);

for i = 1:n    
    for j = 1:n
        if i~=j 
            cur_ht = t(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
            cur_t = nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
            lambda(i,j) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);
        end
    end
end


end

function lambda = lambda_from_admm(nt,t)
n = size(nt,1)/3;

lambda = zeros(n,n,n);
if n >= 70
    parfor i = 1:n    
        for j = 1:n
            for k = 1:n
                if i~=j || j~=k || i~=k
                    cur_ht = tens2mat(t(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                    cur_t = tens2mat(nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                    lambda(i,j,k) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);

    %               lambda(i,j,k) = ((frob(cur_t))^2) / trace(transpose(cur_ht) * cur_t) ;
                end
            end
        end
    end
else
    for i = 1:n    
        for j = 1:n
            for k = 1:n
                if i~=j || j~=k || i~=k
                    cur_ht = tens2mat(t(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                    cur_t = tens2mat(nt(3*(i-1)+1:3*i,3*(j-1)+1:3*j,3*(k-1)+1:3*k),1);
                    lambda(i,j,k) = trace(transpose(cur_t) * cur_ht) /((frob(cur_ht))^2);

    %               lambda(i,j,k) = ((frob(cur_t))^2) / trace(transpose(cur_ht) * cur_t) ;
                end
            end
        end
    end
end

end

