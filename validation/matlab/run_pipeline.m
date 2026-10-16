function run_pipeline(listFile, pipelineDir, frameRate, spatialScale)
% Run the pipeline's processFile on every folder listed in listFile, then keep the
% compared outputs in compared.mat. Folders that already have compared.mat are skipped.
    addpath(fullfile(pipelineDir, 'MATLAB_scripts'));
    keep = {'domFreq', 'domFreqProximal', 'domFreqCentral', 'domFreqDistal', ...
            'legendreCoeffsDomFreqAmp', 'legendreCoeffsDomFreqPhase', 'domFreqAmpSummed', ...
            'phaseLinearityRsq', 'numPhaseCycles', 'averageFlagAngleFromBase', ...
            'domFreqDominance', 'qualityOfFitAngle', 'qualityOfFitXYOptimal', ...
            'amplitudeSpatialAllData', 'amplitudeAngularAllData', 'flagLength', 'badFileFlag'};
    folders = readlines(listFile);
    folders = folders(strlength(folders) > 0);
    for i = 1:numel(folders)
        folder = char(folders(i));
        if isfile(fullfile(folder, 'compared.mat'))
            continue
        end
        fprintf('%d / %d %s\n', i, numel(folders), folder);
        try
            processFile(folder, 'rawcoordinates.txt', frameRate, spatialScale, 0.9, 200);
            S = load(fullfile(folder, 'output.mat'), keep{:});
            save(fullfile(folder, 'compared.mat'), '-struct', 'S', '-v7');
        catch err
            fid = fopen(fullfile(folder, 'failed.txt'), 'w');
            fprintf(fid, '%s', err.message);
            fclose(fid);
        end
        close all force
    end
end
