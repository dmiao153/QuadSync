function core = generate_E_core()
    core = zeros(6,6);
    core(1,6) = 1;
    core(2,5) = 1;
    core(3,4) = -1;
    core(4,3) = -1;
    core(5,2) = 1;
    core(6,1) = -1;
end