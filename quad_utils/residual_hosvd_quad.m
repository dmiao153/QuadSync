function res = residual_hosvd_quad(cur_t, np, t_scale)
    [U,S,sv] = mlsvd(cur_t);
    cs = U{1}(:,1:4);
    B = U{2}(:,1:4);
    C = U{3}(:,1:4);
    D = U{4}(:,1:4);
    G = S(1:4, 1:4, 1:4,1:4);
    cur_t2 = tmprod(G, {cs,B,C,D}, 1:4);


    cur_t2(isnan(cur_t2)) = 0;
    cameras = retrieve_cameras_hosvd(cur_t2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    uncali = false;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [dd2,nn2,Hf, npmat1, npmat2] = compare_projection_matrices(reorder_np_array_to_cell(cameras), np, uncali);
    res2 = dd2 / nn2;

    cameras = normalizenpmat4(cameras);
    np = normalizenpmat4(reorder_np_cell_to_array(np));

    [errR_mean, errR_median, errT_mean, errT_median] = comparison_for_calibrated_cameras(reorder_np_array_to_cell(cameras), reorder_np_array_to_cell(np));
    res.errR_mean = errR_mean;
    res.errR_median = errR_median;
    res.errT_mean = errT_mean / t_scale;
    res.errT_median = errT_median / t_scale;
    res.res2 = res2;
end