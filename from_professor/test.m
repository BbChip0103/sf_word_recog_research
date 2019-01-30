


% load('conv1.mat'); % data load
% 
% disp(size(conv1_weights)); % ctrl+r / ctrl+t

% i=1;
% mat = conv1_weights(:,:,1,i);
% 
% my_range = [-0.1,0.1];
% 
% figure();
% imagesc(mat,my_range);
% colorbar();

my_range = [-0.1,0.1];

figure();

for j=1:8
    subplot(3,3,j)
    imagesc(conv1_weights(:,:,1,j),my_range);
    colormap('gray');
    colorbar();
    
end

