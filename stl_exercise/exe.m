T = evalc('stackedAEExercise');

dateTime = datestr(now);
filename = ['stackedAEExercise.m-' dateTime(1:11) '-' dateTime(13:end) '.txt'];
fileID = fopen(filename, 'w');
fprintf(fileID, T);