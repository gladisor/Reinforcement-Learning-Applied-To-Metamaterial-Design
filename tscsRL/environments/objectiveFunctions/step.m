function [q_, v_] = step(u, q, v, N)
% DYNAMICS Calculates time derivatives of states for 3 sphere collision
% problem within a square area.
% 
% States are: qA, qB, qC
% qA = [xA yA xdA ydA]
% qB = [xA yA xdA ydA]
% qC = [xA yA xdA ydA]
%
% mA, mB, mC : mass of spheres (kg)
% RA, RB, RC : radii of spheres (m)
% cA, cB, cC : aerodynamic drag coeffs. of spheres (N/m/s)
% k          : sphere-sphere contact stiffness (N/m)
% kwall      : sphere-wall contact stiffness (N/m)
% cwall      : sphere-wall contact damping (N/m/s)

% Copyright 2020, The MathWorks Inc.

% Constants
% sphere masses (kg)
m = 1;
a = 1;
delta = 0.1;


% Environment limits
xLimits = [-6 6];
yLimits = [-6 6];

% sphere drag coefficient (N/m/s)
c = 0.1;

% sphere-sphere contact stiffness (N/m)
k = 10;

% sphere-wall contact stiffness (N/m)
kwall = 10;

% sphere-wall contact damping (N/m/s)
cwall = 0.1;



%
t = 0.1;
q = reshape(q', 2, N)';
u = reshape(u', 2, N)';
v = reshape(v', 2, N)';

q_ = zeros(N,2);
v_ = zeros(N,2);
x = zeros(N,1);
y = zeros(N,1);
dx = zeros(N,1);
dy = zeros(N,1);
dist_sphere = zeros(N);
dist_wall = zeros(N,4,1);
coll_sphere = zeros(N);
sf_x = zeros(N);
sf_y = zeros(N);
sf_wallA_x = zeros(N,1);
sf_wallA_y = zeros(N,1);
sf_wallB_x = zeros(N,1);
sf_wallB_y = zeros(N,1);
sf_wallC_x = zeros(N,1);
sf_wallC_y = zeros(N,1);
sf_wallD_x = zeros(N,1);
sf_wallD_y = zeros(N,1);

df_wallA_x = zeros(N,1);
df_wallA_y = zeros(N,1);
df_wallB_x = zeros(N,1);
df_wallB_y = zeros(N,1);
df_wallC_x = zeros(N,1);
df_wallC_y = zeros(N,1);
df_wallD_x = zeros(N,1);
df_wallD_y = zeros(N,1);

sd_force_x = zeros(N,1);
sd_force_y = zeros(N,1);

ddx = zeros(N,1);
ddy = zeros(N,1);



% Unpack state vectors
for i = 1:N
    x(i) = q(i,1);
    y(i) = q(i,2);
    dx(i) = v(i,1);
    dy(i) = v(i,2);
end


% Distance between spheres
for i = 1:N
    x_ = x(i);
    y_ = y(i);
    for j = i+1:N
        dist_sphere(i, j) = sqrt((x_ - x(j))^2 + (y_ - y(j))^2);
        dist_sphere(j, i) = dist_sphere(i, j);
    end
end


% Distance of spheres to wall
% WallA - left wall
% WallB - top wall
% WallC - right wall
% WallD - bottom wall
% If these distances are -ve then it indicates collision
for i = 1:N
    dist_wall(i,1) = (x(i)-a) - xLimits(1);
    dist_wall(i,2) = yLimits(2) - (y(i)+a);
    dist_wall(i,3) = xLimits(2) - (x(i)+a);
    dist_wall(i,4) = (y(i)-a) - yLimits(1);
end


% Collision conditions
% Distance between spheres
for i = 1:N
    for j = i+1:N
        coll_sphere(i, j) = dist_sphere(i,j) < 2*a+delta;
        coll_sphere(j, i) = coll_sphere(i, j);
    end
end

% Spring force between spheres
for i = 1:N
    x_ = x(i);
    y_ = y(i);
    for j = i+1:N
        dist_vector = [x(j);y(j)] - [x_;y_];
        sf = coll_sphere(i,j) * (2*a - dist_sphere(i,j)) * k * (dist_vector./norm(dist_vector));
        sf_x(i, j) = sf(1);
        sf_y(i, j) = sf(2);
        sf_x(j, i) = -sf_x(i, j);
        sf_y(j, i) = -sf_y(i, j);
    end
end


% Spring force between spheres and walls
for i = 1:N
    sfWallA = (dist_wall(i,1) < 0) * abs(dist_wall(i,1)) * kwall * [1;0];
    sfWallB = (dist_wall(i,2) < 0) * abs(dist_wall(i,1)) * kwall * [0;-1];
    sfWallC = (dist_wall(i,3) < 0) * abs(dist_wall(i,1)) * kwall * [-1;0];
    sfWallD = (dist_wall(i,4) < 0) * abs(dist_wall(i,1)) * kwall * [0;1];
    sf_wallA_x(i) = sfWallA(1);
    sf_wallA_y(i) = sfWallA(2);
    sf_wallB_x(i) = sfWallB(1);
    sf_wallB_y(i) = sfWallB(2);
    sf_wallC_x(i) = sfWallC(1);
    sf_wallC_y(i) = sfWallC(2);
    sf_wallD_x(i) = sfWallD(1);
    sf_wallD_y(i) = sfWallD(2);  
end


% Damping force on spheres due to wall
for i = 1:N
    dfWallA = (dist_wall(i,1) < 0) * cwall * dx(i) * [1;0];
    dfWallB = (dist_wall(i,1) < 0) * cwall * dy(i) * [0;-1];
    dfWallC = (dist_wall(i,1) < 0) * cwall * dx(i) * [-1;0];
    dfWallD = (dist_wall(i,1) < 0) * cwall * dy(i) * [0;-1];
    df_wallA_x(i) = dfWallA(1);
    df_wallA_y(i) = dfWallA(2);
    df_wallB_x(i) = dfWallB(1);
    df_wallB_y(i) = dfWallB(2);
    df_wallC_x(i) = dfWallC(1);
    df_wallC_y(i) = dfWallC(2);
    df_wallD_x(i) = dfWallD(1);
    df_wallD_y(i) = dfWallD(2);  
end

% Net spring-damper force on spheres
for i = 1:N
    sd_force_x(i) = sum(sf_x(i,:)) + sf_wallA_x(i) + sf_wallB_x(i) + sf_wallC_x(i) + sf_wallD_x(i) + df_wallA_x(i) + df_wallB_x(i) + df_wallC_x(i) + df_wallD_x(i);
    sd_force_y(i) = sum(sf_y(i,:)) + sf_wallA_y(i) + sf_wallB_y(i) + sf_wallC_y(i) + sf_wallD_y(i) + df_wallA_y(i) + df_wallB_y(i) + df_wallC_y(i) + df_wallD_y(i);
end

% Equations of motion
for i = 1:N
    ddx(i) = (u(i,1) - c * dx(i) + sd_force_x(i)) / m;
    ddy(i) = (u(i,2) - c * dy(i) + sd_force_y(i)) / m;
end


% State derivatives
for i = 1:N
    q_(i, 1) = x(i) + dx(i) * t + 0.5 * ddx(i) * t^2;
    q_(i, 2) = y(i) + dy(i) * t + 0.5 * ddy(i) * t^2;
    v_(i, 1) = dx(i) + ddx(i) * t;
    v_(i, 2) = dy(i) + ddy(i) * t;
end

q_ = reshape(q_', 2*N, 1)';
v_ = reshape(v_', 2*N, 1)';

end