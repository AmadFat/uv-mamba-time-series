clear; clc

train_src_name = "0508/train.mat";
train_tgt_name = "train/0508.mat";

load(train_src_name, "C");
assert(size(C, 1) == 2)

n_typhoons = size(C, 2);
latlons(1, n_typhoons) = struct('LatLon', [], 'Meta', '');

for i = 1: n_typhoons
    lat = C(1, i);
    lat = lat{1}' / 10;
    lon = C(2, i);
    lon = lon{1}' / 10;
    latlon = horzcat(lat, lon);
    latlons(1, i).LatLon = latlon;

    latlons(1, i).Meta = sprintf('{"source_file": "%s"}', "main_table");
end

save(train_tgt_name, "latlons")

clc; clear

test_src_name = "0508/test.mat";
test_tgt_name = "test/0508.mat";

load(test_src_name, "PremS")

cu_nsample = 0;

n_cols = size(PremS, 2);

for i = 1: n_cols
    c = PremS(1, i);
    arr = c{1};
    nsample = size(arr, 1);
    cu_nsample = cu_nsample + nsample;
end

latlons(1, cu_nsample) = struct("LatLon", [], "Velocity", [], "Direction", [], "Meta", "");

sample_idx = 1;

for i = 1: n_cols
    
    c = PremS(1, i);
    arr = c{1};
    nsample = size(arr, 1);

    for j = 1: nsample

        slice = arr(j, :);
        latlon = slice(1: 2)/10;
        velocity = slice(3);
        direction = slice(4);
        latlons(1, sample_idx).LatLon = latlon;
        latlons(1, sample_idx).Velocity = velocity;
        latlons(1, sample_idx).Direction = direction;
        latlons(1, sample_idx).Meta = sprintf('{"source_file": "%s"}', "main_table");
        sample_idx = sample_idx + 1;

    end

end

save(test_tgt_name, "latlons")