% Sample MATLAB Code
% Basic operations and plotting

clc;
clear all;
close all;

%% Basic Math Operations
a = 10;
b = 5;

fprintf('Addition: %d\n', a + b);
fprintf('Subtraction: %d\n', a - b);
fprintf('Multiplication: %d\n', a * b);
fprintf('Division: %.2f\n', a / b);

%% Array Operations
x = 0:0.1:2*pi;
y_sin = sin(x);
y_cos = cos(x);

%% Plotting
figure;
subplot(2,1,1);
plot(x, y_sin, 'b-', 'LineWidth', 2);
title('Sine Wave');
xlabel('x');
ylabel('sin(x)');
grid on;

subplot(2,1,2);
plot(x, y_cos, 'r-', 'LineWidth', 2);
title('Cosine Wave');
xlabel('x');
ylabel('cos(x)');
grid on;

%% Matrix Operations
A = [1 2 3; 4 5 6; 7 8 9];
B = A';  % Transpose

fprintf('\nMatrix A:\n');
disp(A);

fprintf('Transpose of A:\n');
disp(B);

fprintf('Determinant of A: %.2f\n', det(A));

%% Simple Statistics
data = randn(1, 100);  % Random data

fprintf('\nStatistics:\n');
fprintf('Mean: %.4f\n', mean(data));
fprintf('Std Dev: %.4f\n', std(data));
fprintf('Min: %.4f\n', min(data));
fprintf('Max: %.4f\n', max(data));