function [roterror, transerror] = consistency_measure(cycles_cams)
    % cycle_cams is an sptensor of size 9 x 4 x 4, so that the three cameras are concatenated on top of each other
    % and (:,:,i) are the cameras for the ith index in the cycle
    % 123,234,341,412
    
    [Hf1, Psa1] = findHomography({cycles_cams(1:3,:,2),cycles_cams(4:6,:,2)}, ...
            {cycles_cams(4:6,:,1),cycles_cams(7:9,:,1)});
    
    cycles_cams(:,:,2) = cycles_cams(:,:,2) * Hf1;
    
    [Hf2, Psa2] = findHomography({cycles_cams(1:3,:,3),cycles_cams(4:6,:,3)}, ...
            {cycles_cams(4:6,:,2),cycles_cams(7:9,:,2)});
    
    cycles_cams(:,:,3) = cycles_cams(:,:,3) * Hf2;
    
    [Hf3, Psa3] = findHomography({cycles_cams(1:3,:,4),cycles_cams(4:6,:,4)}, ...
            {cycles_cams(4:6,:,3),cycles_cams(7:9,:,3)});
    
    cycles_cams(:,:,4) = cycles_cams(:,:,4) * Hf3;
    
    % now we can compute the consistency measure
    % we can use the trifocal tensor to compute the homography  
    
    % We want to check the new distance between cycles_cams(1:3,:,1) and cycles_cams(4:6,:,4) and also 
    % cycles_cams(4:6,:,1) and cycles_cams(7:9,:,4)
    
    [R1,t1] = project_to_calibrated_cameras(cycles_cams(1:3,:,1)/norm(cycles_cams(1:3,1:3,1)));
    [R2,t2] = project_to_calibrated_cameras(cycles_cams(4:6,:,4)/norm(cycles_cams(4:6,1:3,4)));
    
    [R3,t3] = project_to_calibrated_cameras(cycles_cams(4:6,:,1)/norm(cycles_cams(4:6,1:3,1)));
    [R4,t4] = project_to_calibrated_cameras(cycles_cams(7:9,:,4)/norm(cycles_cams(7:9,1:3,4)));

    [E,e,normE,norme] = CompareRotations(cat(3,R1,R3),cat(3,R2,R4));
    roterror = E(1,1);
    roterror = min(roterror, 180 - roterror);
    
    %normt1 = t1;
    %normt2 = t2 / norm(t2);
    normt3 = t3 / norm(t3);
    normt4 = t4 / norm(t4);
    transerror = min(norm(normt3 - normt4),norm(normt3 + normt4));
end