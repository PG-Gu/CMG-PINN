function run_official_driver(inputPath, outputPath, psiSource, tspSource, runtimeRoot, concordeCommand)
payload = jsondecode(fileread(inputPath));
psiRoot = fullfile(runtimeRoot, 'psi-official');
tspRoot = fullfile(runtimeRoot, 'tsp-official');
mkdir(psiRoot);
mkdir(tspRoot);
copyfile(fullfile(psiSource, 'ProcessValidityIndices.m'), psiRoot);
copyfile(fullfile(psiSource, 'private'), fullfile(psiRoot, 'private'));
copyfile(fullfile(tspSource, 'CommunitySeparability.m'), tspRoot);
copyfile(fullfile(tspSource, 'private'), fullfile(tspRoot, 'private'));
copyfile(fullfile(fileparts(mfilename('fullpath')), 'extract_official_pairwise_aupr.m'), tspRoot);

settingsPath = fullfile(tspRoot, 'LocalSettings.m');
fid = fopen(settingsPath, 'w');
assert(fid ~= -1, 'Cannot write LocalSettings.m');
settingsCleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'runtimeSettings = struct();\n');
fprintf(fid, 'runtimeSettings.rootPath = strcat(pwd, ''/'');\n');
fprintf(fid, 'runtimeSettings.tempPath = strcat(runtimeSettings.rootPath, ''.temp/'');\n');
fprintf(fid, 'createDirectory(runtimeSettings.tempPath);\n');
fprintf(fid, 'runtimeSettings.concordePath = ''%s'';\n', strrep(concordeCommand, '\', '/'));
fprintf(fid, 'if ~isfile(runtimeSettings.concordePath), error(''Concorde command not found''); end\n');
clear settingsCleanup;

addpath(psiRoot);
addpath(tspRoot);
oldDirectory = pwd;
directoryCleanup = onCleanup(@() cd(oldDirectory));
cd(runtimeRoot);
variants = {'cps', 'ldps', 'tsps'};
results = cell(numel(payload.cases), 1);
for ix = 1:numel(payload.cases)
    item = payload.cases(ix);
    labels = cellstr(string(item.group_labels));
    labels = labels(:);
    [counts, positives] = groupcounts(labels);
    [~, largest] = max(counts);
    positives(largest) = [];
    positives = cellstr(string(positives));
    positives = positives(:);
    psi = ProcessValidityIndices(item.coordinates, labels, positives, ...
        'Indices', [1, 7], 'ProjectionType', 'centroid', ...
        'CenterFormula', 'median', 'Trustworthiness', 0);
    entry = struct();
    entry.method = item.method;
    entry.psi_roc = psi.PSIROC;
    entry.psi_pr = psi.PSIPR;
    entry.psi_mcc = psi.PSIMCC;
    for vx = 1:numel(variants)
        variant = variants{vx};
        [measures, metadata] = CommunitySeparability(item.coordinates, labels, variant, ...
            'positives', positives, 'permutations', 0);
        pairwise = extract_official_pairwise_aupr(metadata, positives);
        assert(numel(pairwise) == 6 && all(isfinite(pairwise)), 'Expected six finite group-pair scores');
        assert(abs(mean(pairwise) / (1 + std(pairwise)) - measures.aupr) <= 1e-12, ...
            'AUPR aggregation mismatch');
        entry.([variant, '_aupr']) = measures.aupr;
    end
    results{ix} = entry;
end

fid = fopen(outputPath, 'w');
assert(fid ~= -1, 'Cannot write metric output');
outputCleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(struct('cases', {results})), 'char');
end
