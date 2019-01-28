%
rst_dir = 'model/cnn_6conv_2fc'; disp(rst_dir);

load([rst_dir '/accuracy.mat']);
figure(1);plot([train_acc', valid_acc', test_acc']); legend('train','valid','test');
saveas(gcf,[rst_dir '/accuracy'], 'png')
fprintf('%.2f,\t%.2f,\t%.2f\n',train_acc(end)*100,valid_acc(end)*100,test_acc(end)*100);

