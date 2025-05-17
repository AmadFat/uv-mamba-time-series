clear; clc;

src_name = '0515/train.mat';
tgt_name = 'train/0515.mat';
load(src_name, 'selectedTyphoons');

n_typhoons = size(selectedTyphoons, 2);
latlons(1, n_typhoons) = struct('LatLon', [], 'Meta', '');

for i = 1:n_typhoons
    % 直接复制 LatLon 字段
    latlon = selectedTyphoons(1, i).LatLon;

    latlons(1, i).LatLon = latlon;
    
    % 处理 Name 字段（直接赋值，不使用 {1}）
    name = selectedTyphoons(1, i).Name;
    
    % 处理 SourceFile 字段
    src_file = selectedTyphoons(1, i).SourceFile;
    
    % 检查 Name 是否为空，若为空则设置默认值
    if iscell(name)
        name = name{1};
    end
    % 构建 Meta 字段（使用双引号包裹字符串）
    latlons(1, i).Meta = sprintf('{"name": "%s", "source_file": "%s"}', name, src_file);
end

% 保存新结构体
save(tgt_name, 'latlons');