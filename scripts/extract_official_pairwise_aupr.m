function values = extract_official_pairwise_aupr(metadata, positives)
positives = convertArrayItemsToString(positives);
values = zeros(1, numel(metadata));
for ix = 1:numel(metadata)
    labelsA = repmat({metadata(ix).communityNameGroupA}, size(metadata(ix).dataGroupA, 1), 1);
    labelsB = repmat({metadata(ix).communityNameGroupB}, size(metadata(ix).dataGroupB, 1), 1);
    memberships = [labelsA; labelsB];
    positiveClass = '';
    for px = 1:numel(positives)
        if any(ismember(memberships, positives{px}))
            positiveClass = positives{px};
            break;
        end
    end
    [~, values(ix)] = computeAUCAUPR(memberships, metadata(ix).scores, positiveClass);
end
end
