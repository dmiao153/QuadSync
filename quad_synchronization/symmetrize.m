function S = symmetrize(T)
% generated using chatgpt
dims = size(T);
d = ndims(T);

perms_modes = perms(1:d);
nPerms = size(perms_modes,1);

S = zeros(dims);

for p = 1:nPerms
    perm = perms_modes(p,:);
    S = S + permute(T,perm);
end

S = S / nPerms;
end


