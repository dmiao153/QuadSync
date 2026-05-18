function [localdata, validity] = transform_LUD_data(data,indices)
    localdata = struct;
    tripletindices = sort([3*(indices-1)+1; 3*(indices-1)+2; 3*indices]);
    localdata.AdjMat = data.AdjMat(indices,indices);
    localdata.R = data.R(:,:,indices);
    localdata.t = data.t(:,:,indices);
    localdata.n = length(indices);
    localdata.K = data.K(:,:,indices);
    localdata.Focal_gt = data.Focal_gt(indices);
    localdata.Hmat = data.Hmat(tripletindices, tripletindices);
    localdata.tijGT = data.tijGT(indices,indices);
    localdata.corr = data.corr(indices, indices);
    localdata.G_gt = data.G_gt(tripletindices, tripletindices);
    localdata.E_est = data.E_est(indices, indices);
    localdata.E_gt = data.E_gt(tripletindices, tripletindices);
    localdata.E = data.E(tripletindices, tripletindices);
    localdata.keepedge = ones(1,sum(localdata.AdjMat,'all')/2);
    localdata.goodcount = 0;
    localdata.edgecount = length(localdata.keepedge);
    localdata.W_x_full = zeros(2,localdata.n);
    localdata.W_y_full = zeros(2,localdata.n);
    localdata.W_mask_full = zeros(2,localdata.n);
    localdata.CompInds = indices;

    validity = true;
    for i = 1:length(indices)
        if nnz(localdata.AdjMat(i,:)) == 0
            validity = false;
        end
    end


    for i = 1:length(indices)
        for j = 1:length(indices)
            if localdata.AdjMat(i,j) == 0
                localdata.corr{i,j} = [];
            end
        end
    end

end