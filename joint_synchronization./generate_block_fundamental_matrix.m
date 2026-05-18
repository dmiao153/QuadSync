%%% generates block fundamental matrix. input: np, cell array. n, int. output,
%%% T, 3n x 3n tensor. 
function E = generate_block_fundamental_matrix(np,n)

    P = calculate_stacked_exterior_square(np,n);
    
    E_core = generate_E_core();

    E = P * E_core * P';

end