figure;
for i = 1:231
    %data = training_data(i, :);
    %imshow(reshape(data, 49, 176), [0, 1]);
    %print(num2str(i), '-dpng');
    %w = waitforbuttonpress;
end

mean_data = mean(training_data, 1) + 0.001;
weights = log(mean_data ./ (1 - mean_data));
imshow(reshape(weights, 49, 176), [-7, 1]);

training_scores = training_data * weights';
validation_scores = validation_data * weights';
cheating_scores = cheat_data * weights';

mean(training_scores)
mean(validation_scores)
mean(cheating_scores)

b = -1000:20:0;
a1 = hist(training_scores, b);
a2 = hist(validation_scores, b);
a3 = hist(cheating_scores, b);
plot(b,[a1;a2;a3]);
legend('training', 'validation', 'cheating');