syms x1 x2 x3 x4;
syms y1 y2 y3 y4;

mat = [0, x3 * y4 - x4 * y3, x4 * y2 - x2 * y4, x2 * y3 - x3 * y2;
       x4 * y3 - x3 * y4, 0, x1 * y4 - x4 * y1, x3 * y1 - x1 * y3;
       x2 * y4 - x4 * y2, x4 * y1 - x1 * y4, 0, x1 * y2 - x2 * y1;
       x3 * y2 - x2 * y3, x1 * y3 - x3 * y1, x2 * y1 - x1 * y2, 0];

rref(mat)

rank(mat)