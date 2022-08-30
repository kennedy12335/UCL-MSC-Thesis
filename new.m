clear; close all; clc

%% The following will apply to each section of materials
% After reading the audio file, i manually extract the single impact sounds
% for analysis. This was done by plotting and finding out what are the
% start and end points of sounds. they are usually between the range of 2-3
% seconds.

%After that, the fundamental frequency was found for each sound and stored.
%The audiosample of that particular sample is also store in their respective material folder.
%% For metal%%

metal_freq = [];
[audio, fs] = audioread('Metal/Euston 4.m4a');

t = (0:length(audio) - 1) / fs;

indices = cell(1, 19);
indices{1, 1} = find(t > 2.45 & t < 2.8);
indices{1, 2} = find(t > 4.05 & t < 4.4);
indices{1, 3} = find(t > 5.6 & t < 5.9);
indices{1, 4} = find(t > 7.25 & t < 7.55);
indices{1, 5} = find(t > 8.85 & t < 9.2);
indices{1, 6} = find(t > 10.7 & t < 11);
indices{1, 7} = find(t > 12.55 & t < 12.85);
indices{1, 8} = find(t > 14.25 & t < 14.55);
indices{1, 9} = find(t > 15.9 & t < 16.2);
indices{1, 10} = find(t > 17.55 & t < 17.9);

audio1 = cell(1, 19);

for i = 1:10
    audio1{1, i} = audio(indices{1, i}(1):indices{1, i}(end));

    f0 = pitch(audio1{1, i}, fs);
    metal_freq = [metal_freq; mean(f0)];

    path = append('sound/metal/', 'M', num2str(i), '.wav');
    audiowrite(path, audio1{1, i}, fs)
end

[audio, fs] = audioread('Metal/Euston 5.m4a');

t = (0:length(audio) - 1) / fs;

indices{1, 11} = find(t > 1.75 & t < 2);
indices{1, 12} = find(t >= 3.45 & t < 3.65);
indices{1, 13} = find(t > 5.15 & t < 5.35);
indices{1, 14} = find(t > 6.72 & t < 6.92);
indices{1, 15} = find(t > 8.3 & t < 8.5);
indices{1, 16} = find(t > 9.935 & t < 10.1);
indices{1, 17} = find(t > 11.53 & t < 11.75);
indices{1, 18} = find(t > 13.15 & t < 13.35);
indices{1, 19} = find(t > 14.84 & t < 15);

for i = 11:19
    audio1{1, i} = audio(indices{1, i}(1):indices{1, i}(end));

    f0 = pitch(audio1{1, i}, fs);
    metal_freq = [metal_freq; mean(f0)];

    path = append('sound/metal/', 'M', num2str(i), '.wav');
    audiowrite(path, audio1{1, i}, fs)

end

%% For plastic%%
plastic_freq = [];
[audio, fs] = audioread('plastic/Euston 7.m4a');
% sound(audio*10,fs)
t = (0:length(audio) - 1) / fs;
indices = cell(1, 19);
indices{1, 1} = find(t > 1.35 & t < 1.65);
indices{1, 2} = find(t > 2.75 & t < 3.1);
indices{1, 3} = find(t > 6.25 & t < 6.55);
indices{1, 4} = find(t > 7.75 & t < 8.1);
indices{1, 5} = find(t > 9.2 & t < 9.5);
indices{1, 6} = find(t > 10.65 & t < 10.95);
indices{1, 7} = find(t > 12.05 & t < 12.35);
indices{1, 8} = find(t > 13.5 & t < 13.8);
indices{1, 9} = find(t > 15 & t < 15.3);

audio1 = cell(1, 19);

for i = 1:9
    audio1{1, i} = audio(indices{1, i}(1):indices{1, i}(end));

    f0 = pitch(audio1{1, i}, fs);
    plastic_freq = [plastic_freq; mean(f0)];

    path = append('sound/plastic/', 'PL', num2str(i), '.wav');
    audiowrite(path, audio1{1, i}, fs)

end

[audio, fs] = audioread('plastic/Euston 8.m4a');

t = (0:length(audio) - 1) / fs;

indices{1, 10} = find(t > 4.18 & t < 4.3);
indices{1, 11} = find(t > 5.7 & t < 5.9);
indices{1, 12} = find(t > 7.22 & t < 7.4);
indices{1, 13} = find(t > 8.82 & t < 9);
indices{1, 14} = find(t > 10.35 & t < 10.55);
indices{1, 15} = find(t > 11.86 & t < 12.05);
indices{1, 16} = find(t > 13.4 & t < 13.6);
indices{1, 17} = find(t > 14.86 & t < 15.1);
indices{1, 18} = find(t > 16.61 & t < 16.85);
indices{1, 19} = find(t > 18.21 & t < 18.4);
%
%
for i = 10:19
    audio1{1, i} = audio(indices{1, i}(1):indices{1, i}(end));

    f0 = pitch(audio1{1, i}, fs);
    plastic_freq = [plastic_freq; mean(f0)];

    path = append('sound/plastic/', 'PL', num2str(i), '.wav');
    audiowrite(path, audio1{1, i}, fs)

end

%% For cardboard%%

cardboard_freq = [];

[audio, fs] = audioread('cardboard/Euston.m4a');

t = (0:length(audio) - 1) / fs;

indices = cell(1, 9);
indices{1, 1} = find(t > 1.55 & t < 1.85);
indices{1, 2} = find(t > 2.9 & t < 3.2);
indices{1, 3} = find(t > 4.25 & t < 4.55);
indices{1, 4} = find(t > 5.55 & t < 5.85);
indices{1, 5} = find(t > 6.9 & t < 7.2);
indices{1, 6} = find(t > 8.45 & t < 8.75);
indices{1, 7} = find(t > 9.85 & t < 10.15);
indices{1, 8} = find(t > 11.3 & t < 11.6);
indices{1, 9} = find(t > 13.3 & t < 13.6);

audio1 = cell(1, 17);

for i = 1:9
    audio1{1, i} = audio(indices{1, i}(1):indices{1, i}(end));

    f0 = pitch(audio1{1, i}, fs);
    cardboard_freq = [cardboard_freq; mean(f0)];

    path = append('sound/cardboard/', 'CB', num2str(i), '.wav');
    audiowrite(path, audio1{1, i}, fs)

end

[audio, fs] = audioread('cardboard/Euston 2.m4a');

t = (0:length(audio) - 1) / fs;

indices{1, 10} = find(t > 1.24 & t < 1.4);
indices{1, 11} = find(t > 2.56 & t < 2.75);
indices{1, 12} = find(t > 4 & t < 4.2);
indices{1, 13} = find(t > 5.38 & t < 5.6);
indices{1, 14} = find(t > 6.84 & t < 7);
indices{1, 15} = find(t > 8.35 & t < 8.53);
indices{1, 16} = find(t > 9.74 & t < 9.98);
indices{1, 17} = find(t > 12.18 & t < 12.38);

for i = 10:17
    audio1{1, i} = audio(indices{1, i}(1):indices{1, i}(end));

    f0 = pitch(audio1{1, i}, fs);
    cardboard_freq = [cardboard_freq; mean(f0)];

    path = append('sound/cardboard/', 'CB', num2str(i), '.wav');
    audiowrite(path, audio1{1, i}, fs)

end

%% MERGE Frequency Data%%
metal_freq(:, 2) = 1;
plastic_freq(:, 2) = 2;
cardboard_freq(:, 2) = 3;

freq_list = [metal_freq; plastic_freq; cardboard_freq];

shuffled_list = freq_list(randperm(length(freq_list)))'; %Shuffle the data to remove bias

index = [];

for i = 1:length(shuffled_list)
    index = [index; find(freq_list == shuffled_list(i))];

end

label = [];

for i = 1:length(index)
    label = [label; freq_list(index(i), 2)];
end

result = [];

for i = 1:length(shuffled_list) % Using a threshold method to classify the materials based on the values of their fundamental frequencies

    if (shuffled_list(i) > 200)
        result = [result; 1];
    elseif (shuffled_list(i) < 200 && shuffled_list(i) > 120)
        result = [result; 3];
    else
        result = [result; 2];
    end

end

performance = (sum(label == result) / size(label, 1)) * 100 % performance in the range of 0 to 1

%% Make Dataset

folder = fullfile('sound');
ads = audioDatastore(folder, 'IncludeSubfolders', true, ...
    'FileExtensions', '.wav', ...
    'LabelSource', 'foldernames');

[adsTrain, adsTest] = splitEachLabel(ads, 0.7); % 70 % - 30 % split between training and testing dataset

audioFile = string(ads.Files); % Entire Dataset
[~, ~, Labels] = unique(ads.Labels);
File = table(audioFile, Labels);
writetable(File, 'audio_dataset.csv')

audioFile_train = string(adsTrain.Files); % Training Dataset
[~, ~, Train_Labels] = unique(adsTrain.Labels);
File = table(audioFile_train, Train_Labels);
writetable(File, 'audio_dataset_train.csv')

audioFile_test = string(adsTest.Files); % Testing Dataset.
[~, ~, Test_Labels] = unique(adsTest.Labels);
File = table(audioFile_test, Test_Labels);
writetable(File, 'audio_dataset_test.csv')
