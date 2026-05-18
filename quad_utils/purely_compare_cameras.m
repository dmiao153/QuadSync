function res = purely_compare_cameras(final_cameras, cameras_gt, t_scale)
    uncali = false;
    [dd2,nn2,Hf, npmat1, npmat2] = compare_projection_matrices(reorder_np_array_to_cell(final_cameras), cameras_gt, uncali);
    res2 = dd2 / nn2;

    cameras = normalizenpmat4(final_cameras);
    np = normalizenpmat4(reorder_np_cell_to_array(cameras_gt));

    [errR_mean, errR_median, errT_mean, errT_median] = comparison_for_calibrated_cameras(reorder_np_array_to_cell(cameras), reorder_np_array_to_cell(np));
    res.errR_mean = errR_mean;
    res.errR_median = errR_median;
    res.errT_mean = errT_mean / t_scale;
    res.errT_median = errT_median / t_scale;
    res.res2 = res2;

end